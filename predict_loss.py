from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = Path("models/random_forest_best.joblib")

BASE_COLUMNS = ["Meter Number", "V1", "V2", "V3", "A1", "A2", "A3"]
FEATURE_COLUMNS = [
    "V1",
    "V2",
    "V3",
    "A1",
    "A2",
    "A3",
    "Power",
    "PhaseImb12",
    "PhaseImb23",
    "PhaseImb31",
    "IV_ratio1",
    "IV_ratio2",
    "IV_ratio3",
]

REVIEW_COLUMNS = [
    "MeanVoltage",
    "NominalVoltage",
    "VoltageDeviationPct",
    "VoltageImbalancePct",
    "MeanCurrent",
    "CurrentImbalancePct",
    "LikelyTwoPhaseLineLine",
    "LikelyHalfLoadJumper",
    "VIExpertSeverity",
    "VIExpertStatus",
    "VIExpertReasons",
    "CommitteeLossVotes",
    "CommitteeNormalVetoes",
    "CommitteeDecision",
    "CommitteeReasons",
    "EngineeringDecision",
    "EngineeringReason",
    "FinalPotentialLoss",
]


def load_model(model_path: str | Path = MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file was not found: {model_path}")

    return joblib.load(model_path)


def read_input_file(uploaded_file: BinaryIO | BytesIO, file_name: str) -> pd.DataFrame:
    suffix = Path(file_name).suffix.lower()

    if suffix == ".xlsx":
        return pd.read_excel(uploaded_file, engine="openpyxl")

    if suffix == ".csv":
        return pd.read_csv(uploaded_file)

    raise ValueError("Unsupported file type. Use Excel or CSV.")


def validate_input(df: pd.DataFrame) -> None:
    missing = [column for column in BASE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    validate_input(df)

    prepared = df.copy()
    valid_mask = prepared[BASE_COLUMNS].notna().all(axis=1)
    prepared = prepared.loc[valid_mask].copy()

    numeric_columns = ["V1", "V2", "V3", "A1", "A2", "A3"]
    for column in numeric_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    valid_mask = valid_mask.copy()
    valid_mask.loc[valid_mask] = prepared[numeric_columns].notna().all(axis=1).values
    prepared = prepared.dropna(subset=numeric_columns).copy()

    prepared["Power"] = (
        prepared["V1"] * prepared["A1"]
        + prepared["V2"] * prepared["A2"]
        + prepared["V3"] * prepared["A3"]
    )

    v_mean = prepared[["V1", "V2", "V3"]].mean(axis=1).replace(0, np.nan)

    prepared["PhaseImb12"] = (prepared["V1"] - prepared["V2"]).abs() / v_mean
    prepared["PhaseImb23"] = (prepared["V2"] - prepared["V3"]).abs() / v_mean
    prepared["PhaseImb31"] = (prepared["V3"] - prepared["V1"]).abs() / v_mean

    prepared["IV_ratio1"] = prepared["A1"] / prepared["V1"].replace(0, np.nan)
    prepared["IV_ratio2"] = prepared["A2"] / prepared["V2"].replace(0, np.nan)
    prepared["IV_ratio3"] = prepared["A3"] / prepared["V3"].replace(0, np.nan)

    feature_data = prepared[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    finite_mask = feature_data.notna().all(axis=1)
    prepared = prepared.loc[finite_mask].copy()

    final_valid_mask = pd.Series(False, index=df.index)
    final_valid_mask.loc[prepared.index] = True

    return prepared, final_valid_mask


def predict_loss(
    df: pd.DataFrame,
    threshold: float = 0.5,
    model_path: str | Path = MODEL_PATH,
    model=None,
) -> pd.DataFrame:
    if model is None:
        model = load_model(model_path)

    prepared, final_valid_mask = engineer_features(df)

    results = df.copy()
    results["LossProbability"] = np.nan
    results["PotentialLoss"] = False
    results["AnalysisStatus"] = "Skipped"

    if prepared.empty:
        return results

    probabilities = model.predict_proba(prepared[FEATURE_COLUMNS])[:, 1]
    predictions = probabilities >= threshold

    results.loc[prepared.index, "LossProbability"] = probabilities
    results.loc[prepared.index, "PotentialLoss"] = predictions
    results.loc[prepared.index, "AnalysisStatus"] = "Analyzed"
    results.loc[~final_valid_mask, "AnalysisStatus"] = "Invalid or incomplete readings"

    return results


def engineering_review(
    results: pd.DataFrame,
    enabled: bool = True,
    treat_127v_as_normal: bool = True,
    treat_two_phase_line_line_as_normal: bool = True,
    confirmed_only: bool = True,
    require_vi_confirmed_for_final: bool = True,
    vi_high_can_override_model: bool = True,
    vi_high_auto_confirm: bool = True,
    include_half_load_jumper_suspect: bool = True,
    half_load_jumper_min_probability_pct: float = 99.5,
    use_current_imbalance_as_evidence: bool = False,
    voltage_tolerance_pct: float = 20.0,
    voltage_imbalance_pct: float = 20.0,
    current_imbalance_pct: float = 30.0,
    two_phase_current_similarity_pct: float = 15.0,
    inactive_phase_current_pct: float = 20.0,
    min_active_current: float = 0.2,
    strong_probability_pct: float = 95.0,
    committee_min_votes: int = 9,
) -> pd.DataFrame:
    reviewed = results.copy()

    voltage = reviewed[["V1", "V2", "V3"]].apply(pd.to_numeric, errors="coerce")
    current = reviewed[["A1", "A2", "A3"]].apply(pd.to_numeric, errors="coerce")

    mean_voltage = voltage.mean(axis=1)
    mean_current = current.mean(axis=1)
    nominal_voltage = mean_voltage.apply(_nearest_nominal_voltage)

    reviewed["MeanVoltage"] = mean_voltage
    reviewed["NominalVoltage"] = nominal_voltage
    reviewed["VoltageDeviationPct"] = (
        (mean_voltage - nominal_voltage).abs() / nominal_voltage.replace(0, np.nan) * 100
    )
    reviewed["VoltageImbalancePct"] = (
        voltage.sub(mean_voltage, axis=0).abs().max(axis=1) / mean_voltage.replace(0, np.nan) * 100
    )
    reviewed["MeanCurrent"] = mean_current
    reviewed["CurrentImbalancePct"] = (
        current.sub(mean_current, axis=0).abs().max(axis=1) / mean_current.replace(0, np.nan) * 100
    )
    reviewed["LikelyTwoPhaseLineLine"] = _two_phase_line_line_mask(
        voltage=voltage,
        current=current,
        current_similarity_pct=two_phase_current_similarity_pct,
        inactive_phase_current_pct=inactive_phase_current_pct,
        min_active_current=min_active_current,
    )
    reviewed["LikelyHalfLoadJumper"] = _half_load_jumper_mask(
        voltage=voltage,
        current=current,
        min_active_current=min_active_current,
    )
    vi_review = classify_vi_patterns(voltage=voltage, current=current, nominal_v=230)
    reviewed = pd.concat([reviewed, vi_review], axis=1)
    reviewed.loc[reviewed["LikelyHalfLoadJumper"], "VIExpertSeverity"] = "Medium-High"
    reviewed.loc[reviewed["LikelyHalfLoadJumper"], "VIExpertStatus"] = (
        "شبهة جمبر أو فقد جزئي في حمل إحدى الفازات"
    )
    reviewed.loc[reviewed["LikelyHalfLoadJumper"], "VIExpertReasons"] = (
        "ثلاثة جهود موجودة، فازتان بتيارين متقاربين، والفاز الثالث قريب من نصف حملهما"
    )

    reviewed[["VoltageDeviationPct", "VoltageImbalancePct", "CurrentImbalancePct"]] = reviewed[
        ["VoltageDeviationPct", "VoltageImbalancePct", "CurrentImbalancePct"]
    ].replace([np.inf, -np.inf], np.nan)

    reviewed["EngineeringDecision"] = "لم يصنفه النموذج كفاقد"
    reviewed["EngineeringReason"] = ""
    reviewed["FinalPotentialLoss"] = reviewed["PotentialLoss"].astype(bool)

    model_suspect = reviewed["PotentialLoss"].astype(bool)
    if not enabled:
        reviewed.loc[model_suspect, "EngineeringDecision"] = "فاقد محتمل حسب النموذج"
        reviewed.loc[model_suspect, "EngineeringReason"] = "المراجعة الفنية غير مفعلة"
        return reviewed

    vi_high = reviewed["VIExpertSeverity"].eq("High")
    vi_strong = reviewed["VIExpertSeverity"].isin(["High", "Medium-High"])
    review_candidate = model_suspect | (vi_high_can_override_model & vi_high)

    normal_voltage = reviewed["VoltageDeviationPct"] <= voltage_tolerance_pct
    balanced_voltage = reviewed["VoltageImbalancePct"] <= voltage_imbalance_pct
    balanced_current = reviewed["CurrentImbalancePct"] <= current_imbalance_pct
    strong_probability = reviewed["LossProbability"] * 100 >= strong_probability_pct
    voltage_anomaly = ~normal_voltage | ~balanced_voltage
    current_anomaly = ~balanced_current
    technical_anomaly = voltage_anomaly | (use_current_imbalance_as_evidence & current_anomaly)
    normal_127_operation = (
        treat_127v_as_normal
        & review_candidate
        & (reviewed["NominalVoltage"] == 127.0)
        & normal_voltage
        & balanced_voltage
    )
    normal_two_phase_line_line = (
        treat_two_phase_line_line_as_normal
        & review_candidate
        & reviewed["LikelyTwoPhaseLineLine"]
    )
    half_load_jumper_suspect = (
        include_half_load_jumper_suspect
        & review_candidate
        & reviewed["LikelyHalfLoadJumper"]
        & (reviewed["LossProbability"] * 100 >= half_load_jumper_min_probability_pct)
    )
    committee = _committee_review(
        reviewed=reviewed,
        review_candidate=review_candidate,
        model_suspect=model_suspect,
        strong_probability=strong_probability,
        voltage_anomaly=voltage_anomaly,
        current_anomaly=current_anomaly,
        normal_127_operation=normal_127_operation,
        normal_two_phase_line_line=normal_two_phase_line_line,
        half_load_jumper_suspect=half_load_jumper_suspect,
        vi_high=vi_high,
        vi_strong=vi_strong,
        voltage_tolerance_pct=voltage_tolerance_pct,
        voltage_imbalance_pct=voltage_imbalance_pct,
        strong_probability_pct=strong_probability_pct,
        use_current_imbalance_as_evidence=use_current_imbalance_as_evidence,
        committee_min_votes=committee_min_votes,
    )
    reviewed = pd.concat([reviewed, committee], axis=1)

    if confirmed_only:
        if require_vi_confirmed_for_final:
            technically_supported = review_candidate & (vi_high | half_load_jumper_suspect) & (
                reviewed["CommitteeNormalVetoes"] == 0
            )
            if not vi_high_auto_confirm:
                technically_supported = technically_supported & (
                    (reviewed["CommitteeLossVotes"] >= committee_min_votes) | half_load_jumper_suspect
                )
        else:
            technically_supported = (
                review_candidate
                & (reviewed["CommitteeLossVotes"] >= committee_min_votes)
                & (reviewed["CommitteeNormalVetoes"] == 0)
            )
        technically_normal = review_candidate & ~technically_supported
    else:
        technically_normal = (
            review_candidate
            & normal_voltage
            & balanced_voltage
            & ((~strong_probability & balanced_current) | normal_127_operation | normal_two_phase_line_line)
        )
        technically_supported = review_candidate & ~technically_normal

    technically_supported = review_candidate & ~technically_normal
    if confirmed_only and require_vi_confirmed_for_final:
        technically_supported = technically_supported & (vi_high | half_load_jumper_suspect)
        if not vi_high_auto_confirm:
            technically_supported = technically_supported & (
                (reviewed["CommitteeLossVotes"] >= committee_min_votes) | half_load_jumper_suspect
            )

    reviewed.loc[technically_normal, "FinalPotentialLoss"] = False
    reviewed.loc[technically_supported, "FinalPotentialLoss"] = True
    reviewed.loc[technically_normal, "EngineeringDecision"] = "مستبعد فنيا كحالة طبيعية"
    reviewed.loc[technically_normal, "EngineeringReason"] = (
        "لا توجد قرينة فنية كافية بعد المراجعة الصارمة"
    )
    reviewed.loc[normal_127_operation, "EngineeringReason"] = (
        "جهد 127V ضمن النطاق الاسمي واتزان الجهد ضمن الحدود، لذلك لا يوجد دليل فني مستقل عن النموذج"
    )
    reviewed.loc[normal_two_phase_line_line, "EngineeringReason"] = (
        "نمط فني طبيعي محتمل: تغذية حار-حار على فازين بتيارين متقاربين وفازة ثالثة شبه معدومة"
    )

    reviewed.loc[technically_supported, "EngineeringDecision"] = "فاقد محتمل مؤيد فنيا"
    reviewed.loc[technically_supported, "EngineeringReason"] = reviewed.loc[
        technically_supported
    ].apply(
        _engineering_reason,
        axis=1,
        args=(
            voltage_tolerance_pct,
            voltage_imbalance_pct,
            current_imbalance_pct,
            strong_probability_pct,
            use_current_imbalance_as_evidence,
        ),
    )
    reviewed.loc[review_candidate, "EngineeringReason"] = reviewed.loc[
        review_candidate, "CommitteeReasons"
    ]
    reviewed.loc[technically_supported, "EngineeringDecision"] = reviewed.loc[
        technically_supported, "CommitteeDecision"
    ]
    reviewed.loc[technically_normal & review_candidate, "EngineeringDecision"] = reviewed.loc[
        technically_normal & review_candidate, "CommitteeDecision"
    ]

    return reviewed


def _committee_review(
    reviewed: pd.DataFrame,
    review_candidate: pd.Series,
    model_suspect: pd.Series,
    strong_probability: pd.Series,
    voltage_anomaly: pd.Series,
    current_anomaly: pd.Series,
    normal_127_operation: pd.Series,
    normal_two_phase_line_line: pd.Series,
    half_load_jumper_suspect: pd.Series,
    vi_high: pd.Series,
    vi_strong: pd.Series,
    voltage_tolerance_pct: float,
    voltage_imbalance_pct: float,
    strong_probability_pct: float,
    use_current_imbalance_as_evidence: bool,
    committee_min_votes: int,
) -> pd.DataFrame:
    loss_probability_pct = reviewed["LossProbability"] * 100
    high_confidence_pct = max(95.0, strong_probability_pct)

    votes = {
        "خبير تعلم آلي": model_suspect & strong_probability,
        "مهندس الجهد والتوزيع": review_candidate & (reviewed["VoltageDeviationPct"] > voltage_tolerance_pct),
        "خبير جودة الطاقة": review_candidate & (reviewed["VoltageImbalancePct"] > voltage_imbalance_pct),
        "مهندس التشخيص الفني": review_candidate & (
            vi_strong | half_load_jumper_suspect | (strong_probability & voltage_anomaly)
        ),
        "مهندس العدادات": review_candidate & ~normal_127_operation,
        "خبير الشبكات القديمة": review_candidate & ~normal_two_phase_line_line,
        "خبير الأحمال": review_candidate & (
            current_anomaly
            if use_current_imbalance_as_evidence
            else (voltage_anomaly | vi_high | half_load_jumper_suspect)
        ),
        "مدقق البيانات": review_candidate & (reviewed["AnalysisStatus"].eq("Analyzed") | vi_high),
        "المدقق الصارم": review_candidate
        & ((loss_probability_pct >= high_confidence_pct) | vi_high)
        & (voltage_anomaly | vi_high | half_load_jumper_suspect),
        "رئيس اللجنة": (
            review_candidate
            & (strong_probability | vi_high)
            & (voltage_anomaly | vi_high | half_load_jumper_suspect)
            & ~normal_127_operation
            & ~normal_two_phase_line_line
        ),
    }
    vote_frame = pd.DataFrame(votes, index=reviewed.index).fillna(False)

    vetoes = {
        "اعتراض 127V طبيعي": normal_127_operation,
        "اعتراض حار-حار طبيعي": normal_two_phase_line_line,
    }
    veto_frame = pd.DataFrame(vetoes, index=reviewed.index).fillna(False)

    output = pd.DataFrame(index=reviewed.index)
    output["CommitteeLossVotes"] = vote_frame.sum(axis=1).astype(int)
    output["CommitteeNormalVetoes"] = veto_frame.sum(axis=1).astype(int)
    output["CommitteeDecision"] = "لم يعرض على اللجنة"

    suspected = review_candidate.fillna(False)
    vetoed = suspected & (output["CommitteeNormalVetoes"] > 0)
    accepted = suspected & (output["CommitteeLossVotes"] >= committee_min_votes) & ~vetoed
    deferred = suspected & ~accepted & ~vetoed

    output.loc[vetoed, "CommitteeDecision"] = "مستبعد باعتراض فني"
    output.loc[accepted, "CommitteeDecision"] = "معتمد من لجنة 10 خبراء"
    output.loc[deferred, "CommitteeDecision"] = "غير مؤكد ويحتاج قرينة إضافية"

    output["CommitteeReasons"] = ""
    active_indices = output.index[suspected]
    for row_index in active_indices:
        loss_experts = [name for name, value in vote_frame.loc[row_index].items() if value]
        veto_experts = [name for name, value in veto_frame.loc[row_index].items() if value]

        parts = [f"أصوات التأييد {output.at[row_index, 'CommitteeLossVotes']}/10"]
        if loss_experts:
            parts.append("المؤيدون: " + "، ".join(loss_experts))
        if veto_experts:
            parts.append("اعتراضات: " + "، ".join(veto_experts))
        if output.at[row_index, "CommitteeLossVotes"] < committee_min_votes and not veto_experts:
            parts.append("التوصية: لا تعتمد كفاقد مؤكد إلا بعد قرينة ميدانية أو رفع عتبة الثقة")

        output.at[row_index, "CommitteeReasons"] = " | ".join(parts)

    return output


def _nearest_nominal_voltage(mean_voltage: float) -> float:
    if pd.isna(mean_voltage):
        return np.nan

    nominal_options = [127.0, 220.0, 380.0]
    return min(nominal_options, key=lambda nominal: abs(mean_voltage - nominal))


def _two_phase_line_line_mask(
    voltage: pd.DataFrame,
    current: pd.DataFrame,
    current_similarity_pct: float,
    inactive_phase_current_pct: float,
    min_active_current: float,
) -> pd.Series:
    current_values = current[["A1", "A2", "A3"]].abs().to_numpy(dtype=float)
    voltage_values = voltage[["V1", "V2", "V3"]].abs().to_numpy(dtype=float)

    v_dead = 46.0
    voltage_present = voltage_values >= 105
    voltage_dead = voltage_values <= v_dead
    exactly_two_voltages_present = voltage_present.sum(axis=1) == 2
    exactly_one_voltage_dead = voltage_dead.sum(axis=1) == 1

    active_voltage_index = np.argsort(voltage_values, axis=1)[:, 1:3]
    inactive_voltage_index = np.argsort(voltage_values, axis=1)[:, 0]
    active_voltage = np.take_along_axis(voltage_values, active_voltage_index, axis=1)
    active_current = np.take_along_axis(current_values, active_voltage_index, axis=1)
    inactive_current = current_values[np.arange(len(current_values)), inactive_voltage_index]

    active_voltage_avg = np.nanmean(active_voltage, axis=1)
    active_voltage_diff_pct = (
        np.nanmax(np.abs(active_voltage - active_voltage_avg[:, None]), axis=1)
        / np.where(active_voltage_avg == 0, np.nan, active_voltage_avg)
        * 100
    )
    active_current_avg = np.nanmean(active_current, axis=1)
    active_current_diff_pct = (
        np.nanmax(np.abs(active_current - active_current_avg[:, None]), axis=1)
        / np.where(active_current_avg == 0, np.nan, active_current_avg)
        * 100
    )
    active_current_threshold = np.maximum(min_active_current, active_current.max(axis=1) * inactive_phase_current_pct / 100)
    two_voltage_phases_are_loaded = np.nanmin(active_current, axis=1) >= active_current_threshold
    inactive_phase_current_is_low = inactive_current <= active_current_threshold

    likely_voltage_pair = pd.Series(
        exactly_two_voltages_present
        & exactly_one_voltage_dead
        & (active_voltage_avg >= 105)
        & (active_voltage_avg <= 250)
        & (active_voltage_diff_pct <= 10),
        index=current.index,
    )

    return (
        likely_voltage_pair
        & two_voltage_phases_are_loaded
        & inactive_phase_current_is_low
        & (active_current_diff_pct <= current_similarity_pct)
    ).fillna(False)


def _half_load_jumper_mask(
    voltage: pd.DataFrame,
    current: pd.DataFrame,
    min_active_current: float,
) -> pd.Series:
    voltage_values = voltage[["V1", "V2", "V3"]].abs().to_numpy(dtype=float)
    current_values = current[["A1", "A2", "A3"]].abs().to_numpy(dtype=float)

    all_voltages_present = (voltage_values >= 105).sum(axis=1) == 3
    voltage_avg = np.nanmean(voltage_values, axis=1)
    voltage_spread_pct = (
        np.nanmax(np.abs(voltage_values - voltage_avg[:, None]), axis=1)
        / np.where(voltage_avg == 0, np.nan, voltage_avg)
        * 100
    )
    voltage_reasonably_stable = voltage_spread_pct <= 20

    sorted_current = np.sort(current_values, axis=1)
    low_current = sorted_current[:, 0]
    mid_current = sorted_current[:, 1]
    high_current = sorted_current[:, 2]
    high_pair_avg = (mid_current + high_current) / 2

    high_pair_close_pct = (
        np.abs(high_current - mid_current) / np.where(high_pair_avg == 0, np.nan, high_pair_avg) * 100
    )
    low_to_pair_ratio = low_current / np.where(high_pair_avg == 0, np.nan, high_pair_avg)

    return pd.Series(
        all_voltages_present
        & voltage_reasonably_stable
        & (mid_current >= min_active_current)
        & (high_current >= min_active_current)
        & (low_current >= min_active_current)
        & (high_pair_close_pct <= 15)
        & (low_to_pair_ratio >= 0.35)
        & (low_to_pair_ratio <= 0.65),
        index=current.index,
    ).fillna(False)


def classify_vi_patterns(
    voltage: pd.DataFrame,
    current: pd.DataFrame,
    nominal_v: float = 230,
    min_i: float = 1.0,
    noise_i: float = 0.2,
    high_i: float = 20.0,
) -> pd.DataFrame:
    output = pd.DataFrame(index=voltage.index)
    output["VIExpertSeverity"] = "Normal"
    output["VIExpertStatus"] = "لا يوجد مؤشر فاقد واضح من الفولت والتيار فقط"
    output["VIExpertReasons"] = ""

    phase_map = [("A", "V1", "A1"), ("B", "V2", "A2"), ("C", "V3", "A3")]
    v_dead = 0.20 * nominal_v
    v_very_low = 0.35 * nominal_v
    v_normal = 0.78 * nominal_v

    confirmed_reasons = pd.Series("", index=voltage.index, dtype="object")
    strong_reasons = pd.Series("", index=voltage.index, dtype="object")
    weak_reasons = pd.Series("", index=voltage.index, dtype="object")

    for phase_name, voltage_col, current_col in phase_map:
        v = pd.to_numeric(voltage[voltage_col], errors="coerce")
        i = pd.to_numeric(current[current_col], errors="coerce")

        confirmed_missing_voltage = (v < v_dead) & (i >= min_i)
        confirmed_low_voltage_high_current = (v < v_very_low) & (i >= 5)
        strong_low_voltage_current = (v < v_very_low) & (i >= min_i) & ~confirmed_low_voltage_high_current

        confirmed_reasons = _append_reason(
            confirmed_reasons,
            confirmed_missing_voltage,
            f"الفاز {phase_name}: تيار موجود مع فولت مفقود/شبه صفر",
        )
        confirmed_reasons = _append_reason(
            confirmed_reasons,
            confirmed_low_voltage_high_current,
            f"الفاز {phase_name}: فولت منخفض جدًا مع تيار عالي",
        )
        strong_reasons = _append_reason(
            strong_reasons,
            strong_low_voltage_current,
            f"الفاز {phase_name}: فولت منخفض جدًا مع تيار موجود",
        )

    all_v_dead = (voltage[["V1", "V2", "V3"]] < v_dead).all(axis=1)
    any_i_present = (current[["A1", "A2", "A3"]] >= min_i).any(axis=1)
    confirmed_reasons = _append_reason(
        confirmed_reasons,
        all_v_dead & any_i_present,
        "كل الفولتات مفقودة أو شبه صفر مع وجود تيار",
    )

    normal_voltage_count = (voltage[["V1", "V2", "V3"]] >= v_normal).sum(axis=1)
    zero_current_count = (current[["A1", "A2", "A3"]] < noise_i).sum(axis=1)
    loaded_count = (current[["A1", "A2", "A3"]] >= min_i).sum(axis=1)
    high_loaded_count = (current[["A1", "A2", "A3"]] >= high_i).sum(axis=1)

    strong_reasons = _append_reason(
        strong_reasons,
        (normal_voltage_count == 3) & (zero_current_count == 1) & (loaded_count >= 2),
        "الفولتات طبيعية وفاز واحد تياره صفر تقريبًا والباقي محمل",
    )
    strong_reasons = _append_reason(
        strong_reasons,
        (normal_voltage_count == 3) & (zero_current_count == 2) & (high_loaded_count == 1),
        "الفولتات طبيعية وفاز واحد عليه تيار عالي وفازان صفر تقريبًا",
    )
    weak_reasons = _append_reason(
        weak_reasons,
        (normal_voltage_count == 3) & (zero_current_count == 3),
        "الفولتات طبيعية وكل التيارات صفر تقريبًا؛ لا يثبت فاقد بدون سياق خارجي",
    )

    current_values = current[["A1", "A2", "A3"]].where(current[["A1", "A2", "A3"]] >= min_i)
    max_i = current_values.max(axis=1)
    min_present_i = current_values.min(axis=1)
    active_count = current_values.notna().sum(axis=1)
    weak_reasons = _append_reason(
        weak_reasons,
        (active_count >= 2) & (min_present_i > 0) & ((max_i / min_present_i) >= 10),
        "عدم اتزان تيارات شديد بين الفازات النشطة",
    )

    confirmed = confirmed_reasons.ne("")
    strong = strong_reasons.ne("")
    weak = weak_reasons.ne("")

    output.loc[confirmed, "VIExpertSeverity"] = "High"
    output.loc[confirmed, "VIExpertStatus"] = "فاقد مؤكد أو خلل قياس مؤكد مؤثر على الاحتساب"
    output.loc[confirmed, "VIExpertReasons"] = confirmed_reasons[confirmed]

    strong_only = ~confirmed & strong
    output.loc[strong_only, "VIExpertSeverity"] = "Medium-High"
    output.loc[strong_only, "VIExpertStatus"] = "فاقد محتمل قوي يحتاج تفتيش أو تحقق زمني"
    output.loc[strong_only, "VIExpertReasons"] = strong_reasons[strong_only]

    weak_only = ~confirmed & ~strong & weak
    output.loc[weak_only, "VIExpertSeverity"] = "Low-Medium"
    output.loc[weak_only, "VIExpertStatus"] = "مؤشر ضعيف أو غير كافٍ للحكم من V/I فقط"
    output.loc[weak_only, "VIExpertReasons"] = weak_reasons[weak_only]

    return output


def _append_reason(reasons: pd.Series, mask: pd.Series, reason: str) -> pd.Series:
    mask = mask.fillna(False)
    updated = reasons.copy()
    updated.loc[mask & updated.ne("")] = updated.loc[mask & updated.ne("")] + "؛ " + reason
    updated.loc[mask & updated.eq("")] = reason
    return updated


def _engineering_reason(
    row: pd.Series,
    voltage_tolerance_pct: float,
    voltage_imbalance_pct: float,
    current_imbalance_pct: float,
    strong_probability_pct: float,
    use_current_imbalance_as_evidence: bool,
) -> str:
    reasons = []

    if row["LossProbability"] * 100 >= strong_probability_pct:
        reasons.append(f"احتمال النموذج مرتفع >= {strong_probability_pct:.0f}%")
    if row["VoltageDeviationPct"] > voltage_tolerance_pct:
        reasons.append(f"انحراف الجهد > {voltage_tolerance_pct:.0f}%")
    if row["VoltageImbalancePct"] > voltage_imbalance_pct:
        reasons.append(f"عدم اتزان الجهد > {voltage_imbalance_pct:.1f}%")
    if use_current_imbalance_as_evidence and row["CurrentImbalancePct"] > current_imbalance_pct:
        reasons.append(f"عدم اتزان التيار > {current_imbalance_pct:.0f}%")

    return "، ".join(reasons) if reasons else "مؤيد من النموذج"


def build_excel(results: pd.DataFrame, threshold: float) -> bytes:
    final_suspected = results[results.get("FinalPotentialLoss", results["PotentialLoss"]) == True].copy()
    model_suspected = results[results["PotentialLoss"] == True].copy()
    filtered_normal = results[
        (results["PotentialLoss"] == True)
        & (results.get("FinalPotentialLoss", results["PotentialLoss"]) == False)
    ].copy()

    summary = pd.DataFrame(
        [
            {"Metric": "Threshold", "Value": threshold},
            {"Metric": "Total rows", "Value": len(results)},
            {"Metric": "Analyzed rows", "Value": int((results["AnalysisStatus"] == "Analyzed").sum())},
            {"Metric": "Model potential loss", "Value": len(model_suspected)},
            {"Metric": "Final potential loss", "Value": len(final_suspected)},
            {"Metric": "Technically filtered normal", "Value": len(filtered_normal)},
        ]
    )

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        final_suspected.to_excel(writer, index=False, sheet_name="Final Potential Loss")
        filtered_normal.to_excel(writer, index=False, sheet_name="Filtered Normal")
        model_suspected.to_excel(writer, index=False, sheet_name="Model Suspects")
        results.to_excel(writer, index=False, sheet_name="All Results")
        summary.to_excel(writer, index=False, sheet_name="Summary")
        build_committee_summary(results).to_excel(writer, index=False, sheet_name="Committee Summary")

    return output.getvalue()


def build_committee_summary(results: pd.DataFrame) -> pd.DataFrame:
    model_suspects = results[results["PotentialLoss"] == True]
    final_suspects = results[results.get("FinalPotentialLoss", results["PotentialLoss"]) == True]
    filtered = results[
        (results["PotentialLoss"] == True)
        & (results.get("FinalPotentialLoss", results["PotentialLoss"]) == False)
    ]

    if "CommitteeNormalVetoes" in results.columns:
        vetoed = filtered[filtered["CommitteeNormalVetoes"] > 0]
        deferred = filtered[filtered["CommitteeNormalVetoes"] == 0]
    else:
        vetoed = filtered.iloc[0:0]
        deferred = filtered

    line_line_filtered = filtered[
        filtered.get("LikelyTwoPhaseLineLine", pd.Series(False, index=filtered.index)) == True
    ]
    half_load_suspects = results[
        results.get("LikelyHalfLoadJumper", pd.Series(False, index=results.index)) == True
    ]
    vi_counts = (
        results["VIExpertSeverity"].value_counts().to_dict()
        if "VIExpertSeverity" in results.columns
        else {}
    )

    rows = [
        {"Item": "Model suspects", "Value": len(model_suspects)},
        {"Item": "Final confirmed suspects", "Value": len(final_suspects)},
        {"Item": "Technically filtered", "Value": len(filtered)},
        {"Item": "Filtered by normal veto", "Value": len(vetoed)},
        {"Item": "Deferred for weak evidence", "Value": len(deferred)},
        {"Item": "Filtered line-line two-phase pattern", "Value": len(line_line_filtered)},
        {"Item": "Half-load jumper suspects", "Value": len(half_load_suspects)},
        {"Item": "VI expert High", "Value": vi_counts.get("High", 0)},
        {"Item": "VI expert Medium-High", "Value": vi_counts.get("Medium-High", 0)},
        {"Item": "VI expert Low-Medium", "Value": vi_counts.get("Low-Medium", 0)},
    ]

    if "CommitteeLossVotes" in model_suspects.columns and not model_suspects.empty:
        rows.append(
            {
                "Item": "Average committee votes for model suspects",
                "Value": round(float(model_suspects["CommitteeLossVotes"].mean()), 2),
            }
        )

    recommendations = _committee_recommendations(
        model_count=len(model_suspects),
        final_count=len(final_suspects),
        filtered_count=len(filtered),
        veto_count=len(vetoed),
        deferred_count=len(deferred),
        line_line_count=len(line_line_filtered),
    )
    rows.extend({"Item": "Recommendation", "Value": recommendation} for recommendation in recommendations)

    return pd.DataFrame(rows)


def _committee_recommendations(
    model_count: int,
    final_count: int,
    filtered_count: int,
    veto_count: int,
    deferred_count: int,
    line_line_count: int,
) -> list[str]:
    recommendations = []

    if model_count == 0:
        return ["لا توجد حالات مشتبه بها من النموذج في هذا التشغيل."]

    final_rate = final_count / model_count
    filtered_rate = filtered_count / model_count

    if final_rate > 0.25:
        recommendations.append("النتائج النهائية ما زالت كثيرة: ارفع احتمال النموذج أو ارفع أقل أصوات لاعتماد الفاقد.")
    elif final_rate < 0.03:
        recommendations.append("النتائج النهائية قليلة ومركزة؛ مناسبة لتقليل الزيارات ورفع دقة التفتيش.")
    else:
        recommendations.append("نسبة الحالات النهائية مناسبة كبداية للتفتيش الميداني عالي الأولوية.")

    if line_line_count > 0:
        recommendations.append("استمر في استبعاد نمط حار-حار على فازين لأنه يزيل حالات تشغيل طبيعية محتملة.")

    if veto_count > deferred_count:
        recommendations.append("أغلب الاستبعاد بسبب اعتراضات فنية؛ راجع أنماط التغذية الطبيعية قبل اعتبارها فاقدًا.")

    if deferred_count > final_count:
        recommendations.append("هناك حالات كثيرة غير مؤكدة؛ لا تعتمدها كفاقد إلا بعد صور ميدانية أو قراءة تاريخية.")

    if filtered_rate > 0.70:
        recommendations.append("النموذج يعطي اشتباهًا واسعًا؛ يوصى بإعادة تدريبه ببيانات سليمة ممثلة لجهد 127V وحار-حار.")

    return recommendations
