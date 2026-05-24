from __future__ import annotations

from datetime import datetime
from html import escape
from io import BytesIO

import pandas as pd
import streamlit as st

from predict_loss import engineering_review, load_model, predict_loss, read_input_file


st.set_page_config(
    page_title="تحليل الفاقد المحتمل",
    layout="wide",
)


ANALYSIS_SETTINGS = {
    "threshold": 0.95,
    "enabled": True,
    "treat_127v_as_normal": True,
    "treat_two_phase_line_line_as_normal": True,
    "confirmed_only": True,
    "require_vi_confirmed_for_final": True,
    "vi_high_can_override_model": True,
    "vi_high_auto_confirm": True,
    "include_half_load_jumper_suspect": True,
    "half_load_jumper_min_probability_pct": 99.5,
    "use_current_imbalance_as_evidence": False,
    "voltage_tolerance_pct": 20.0,
    "voltage_imbalance_pct": 20.0,
    "current_imbalance_pct": 30.0,
    "two_phase_current_similarity_pct": 15.0,
    "inactive_phase_current_pct": 20.0,
    "strong_probability_pct": 95.0,
    "committee_min_votes": 9,
}


INPUT_COLUMNS = [
    ("Meter Number", "رقم العداد أو معرف الأصل", "MMF202080000001"),
    ("V1", "جهد الفاز الأول", "230.0"),
    ("V2", "جهد الفاز الثاني", "229.5"),
    ("V3", "جهد الفاز الثالث", "231.0"),
    ("A1", "تيار الفاز الأول", "12.4"),
    ("A2", "تيار الفاز الثاني", "11.9"),
    ("A3", "تيار الفاز الثالث", "12.1"),
]


st.markdown(
    """
    <style>
        .stApp {
            direction: rtl;
            text-align: right;
            background:
                linear-gradient(90deg, rgba(40, 87, 79, 0.045) 1px, transparent 1px),
                linear-gradient(180deg, rgba(40, 87, 79, 0.04) 1px, transparent 1px),
                #f3f6f7;
            background-size: 28px 28px;
            color: #1e2420;
        }
        html, body, [class*="css"] {
            font-family: "Segoe UI", Tahoma, Arial, sans-serif;
        }
        [data-testid="stHeader"] {
            background: rgba(243, 246, 247, 0.88);
        }
        [data-testid="stToolbar"] {
            display: none;
        }
        [data-testid="stSidebar"] {
            display: none;
        }
        .block-container {
            padding-top: 0.75rem;
            padding-bottom: 2rem;
            max-width: 1220px;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        h1 {
            font-size: 2.35rem;
            margin: 0 0 0.35rem 0;
            color: #0f1f21;
            font-weight: 750;
        }
        .app-subtitle {
            color: #51605c;
            margin: 0;
            font-size: 1.02rem;
            line-height: 1.8;
        }
        .hero {
            position: relative;
            overflow: hidden;
            background:
                linear-gradient(135deg, rgba(255,255,255,0.98), rgba(248,251,250,0.96)),
                linear-gradient(90deg, rgba(41, 113, 101, 0.08), rgba(20, 41, 47, 0.04));
            border: 1px solid #d5e0dc;
            border-radius: 8px;
            padding: 1.25rem 1.35rem;
            box-shadow: 0 10px 26px rgba(27, 43, 40, 0.07);
            margin-bottom: 1rem;
        }
        .hero::before {
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(90deg, transparent 0 20%, rgba(46,111,100,0.08) 20% 20.15%, transparent 20.15% 100%),
                linear-gradient(180deg, transparent 0 58%, rgba(46,111,100,0.07) 58% 58.2%, transparent 58.2% 100%);
            pointer-events: none;
        }
        .hero::after {
            content: "";
            position: absolute;
            top: 0;
            right: -30%;
            width: 30%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(232, 169, 73, 0.13), transparent);
            animation: scan-line 5.5s ease-in-out infinite;
            pointer-events: none;
        }
        @keyframes scan-line {
            0%, 30% { right: -32%; opacity: 0; }
            45%, 70% { opacity: 1; }
            100% { right: 102%; opacity: 0; }
        }
        .hero-inner {
            position: relative;
            display: grid;
            grid-template-columns: minmax(0, 1.8fr) minmax(260px, 0.8fr);
            gap: 1rem;
            align-items: center;
        }
        .hero-panel {
            border: 1px solid #d7e4df;
            background: #f7fbfa;
            border-radius: 8px;
            padding: 0.85rem;
            box-shadow: inset 4px 0 0 #e8a949;
        }
        .panel-row {
            display: flex;
            justify-content: space-between;
            gap: 0.8rem;
            border-bottom: 1px solid #e0e8e4;
            padding: 0.42rem 0;
            color: #4e5d58;
            font-size: 0.88rem;
        }
        .panel-row span {
            text-align: right;
        }
        .panel-row:last-child {
            border-bottom: 0;
        }
        .panel-row strong {
            color: #173a36;
            font-weight: 750;
        }
        .workflow-card {
            background: rgba(255,255,255,0.96);
            border: 1px solid #d5e0dc;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            min-height: 5rem;
            margin: 0 0 0.55rem 0;
            box-shadow: 0 4px 14px rgba(27, 43, 40, 0.045);
        }
        .workflow-title {
            color: #203b37;
            font-weight: 700;
            font-size: 0.95rem;
            margin-bottom: 0.45rem;
        }
        .workflow-note {
            color: #63706b;
            font-size: 0.88rem;
            line-height: 1.7;
            margin-bottom: 0.65rem;
        }
        .status-pill {
            display: inline-block;
            background: #e5f2ee;
            color: #1f5d53;
            border: 1px solid #bed9d1;
            border-radius: 999px;
            padding: 0.25rem 0.65rem;
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 0.55rem;
        }
        .status-pill::before {
            content: "";
            display: inline-block;
            width: 0.45rem;
            height: 0.45rem;
            margin-left: 0.45rem;
            border-radius: 999px;
            background: #e8a949;
            box-shadow: 0 0 0 4px rgba(232, 169, 73, 0.16);
        }
        .metric-card {
            background: #ffffff;
            border: 1px solid #d7e0dc;
            border-radius: 8px;
            padding: 1rem 1.05rem;
            box-shadow: 0 4px 14px rgba(27, 43, 40, 0.045);
            min-height: 7.1rem;
            position: relative;
            overflow: hidden;
        }
        .metric-card::before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 4px;
            background: #2e6f64;
        }
        .metric-label {
            color: #53605a;
            font-size: 0.88rem;
            margin-bottom: 0.55rem;
        }
        .metric-value {
            color: #123d39;
            font-weight: 750;
            font-size: 2rem;
            line-height: 1.1;
        }
        .metric-note {
            color: #7a8581;
            font-size: 0.78rem;
            margin-top: 0.5rem;
        }
        .stDownloadButton > button {
            border-radius: 8px;
            border: 1px solid #2e6f64;
            background: #2e6f64;
            color: white;
            min-height: 2.7rem;
            font-weight: 700;
        }
        .stDownloadButton > button:hover {
            border-color: #244f49;
            background: #244f49;
            color: white;
        }
        [data-testid="stFileUploader"] {
            background: #f8faf9;
            border: 1px dashed #b7c6bf;
            border-radius: 8px;
            padding: 0.7rem;
        }
        [data-testid="stFileUploaderDropzone"] {
            min-height: 5.2rem;
            border-radius: 8px;
            background: #eef3f1;
            direction: rtl;
            padding: 1rem;
        }
        [data-testid="stFileUploaderDropzone"] svg {
            width: 1.15rem;
            height: 1.15rem;
        }
        [data-testid="stFileUploaderDropzone"] button {
            border-radius: 8px;
        }
        [data-testid="stFileUploaderFile"] {
            direction: rtl;
            background: #ffffff;
            border: 1px solid #d7e0dc;
            border-radius: 8px;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid #d7e0dc;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 14px rgba(27, 43, 40, 0.045);
        }
        .stAlert {
            border-radius: 8px;
        }
        .app-footer {
            color: #6b7772;
            border-top: 1px solid #dbe3dd;
            font-size: 0.85rem;
            margin-top: 1.4rem;
            padding-top: 0.9rem;
            text-align: center;
        }
        .section-title {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            margin: 0.2rem 0 0.6rem 0;
        }
        .section-title h3 {
            margin: 0;
            color: #1f312e;
            font-size: 1.15rem;
        }
        .section-title span {
            color: #687670;
            font-size: 0.88rem;
        }
        .results-shell {
            background: rgba(255,255,255,0.96);
            border: 1px solid #d7e0dc;
            border-radius: 8px;
            padding: 0.95rem;
            box-shadow: 0 8px 24px rgba(27, 43, 40, 0.055);
        }
        .signal-strip {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.7rem;
            margin: 0.35rem 0 0.85rem 0;
        }
        .signal-item {
            background: #f7fbfa;
            border: 1px solid #dbe7e2;
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            color: #52615c;
            font-size: 0.86rem;
        }
        .signal-item strong {
            display: block;
            color: #173a36;
            font-size: 0.96rem;
            margin-bottom: 0.25rem;
        }
        @media (max-width: 860px) {
            .hero-inner {
                grid-template-columns: 1fr;
            }
            .signal-strip {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    return load_model()


def build_input_template() -> bytes:
    example_data = pd.DataFrame(
        [
            {
                "Meter Number": "METER-001",
                "V1": 230.0,
                "V2": 229.5,
                "V3": 231.0,
                "A1": 12.4,
                "A2": 11.9,
                "A3": 12.1,
            },
            {
                "Meter Number": "METER-002",
                "V1": 127.0,
                "V2": 126.5,
                "V3": 127.4,
                "A1": 4.2,
                "A2": 4.0,
                "A3": 0.1,
            },
        ]
    )
    columns_info = pd.DataFrame(
        [
            {"Column": column, "Description": description, "Example": example}
            for column, description, example in INPUT_COLUMNS
        ]
    )

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        example_data.to_excel(writer, index=False, sheet_name="Data")
        columns_info.to_excel(writer, index=False, sheet_name="Columns Guide")

    return output.getvalue()


def required_columns_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"العمود": column, "الوصف": description, "مثال": example}
            for column, description, example in INPUT_COLUMNS
        ]
    )


def probability_label(value: float) -> str:
    if pd.isna(value):
        return "قرينة فنية مباشرة"
    return f"{value:.1%}"


def priority_label(row: pd.Series) -> str:
    if str(row.get("VIExpertSeverity", "")) == "High":
        return "حرجة"
    if bool(row.get("LikelyHalfLoadJumper", False)):
        return "عالية"
    return "عالية"


def choose_indicator(row: pd.Series) -> str:
    vi_reason = str(row.get("VIExpertReasons", "") or "").strip()
    engineering_reason = str(row.get("EngineeringReason", "") or "").strip()

    if bool(row.get("LikelyHalfLoadJumper", False)):
        return "شبهة نصف حمل/جمبر: فازتان بتيارين متقاربين والفاز الثالث قريب من نصف الحمل."

    if vi_reason:
        return vi_reason

    if engineering_reason:
        return engineering_reason

    return "مؤشرات فنية عالية الثقة حسب قواعد التحليل."


def evidence_category(row: pd.Series) -> str:
    if str(row.get("VIExpertSeverity", "")) == "High":
        return "فاقد مؤكد V/I"
    if bool(row.get("LikelyHalfLoadJumper", False)):
        return "شبهة نصف حمل/جمبر"
    return "فاقد محتمل عالي الثقة"


def add_evidence_score(results: pd.DataFrame) -> pd.DataFrame:
    scored = results.copy()
    severity_rank = scored.get("VIExpertSeverity", pd.Series("", index=scored.index)).map(
        {"High": 4, "Medium-High": 3, "Low-Medium": 2, "Normal": 1}
    ).fillna(0)
    probability = scored.get("LossProbability", pd.Series(0, index=scored.index)).fillna(1.0)
    half_load = scored.get("LikelyHalfLoadJumper", pd.Series(False, index=scored.index)).astype(int)
    voltage_deviation = scored.get("VoltageDeviationPct", pd.Series(0, index=scored.index)).fillna(0)
    voltage_imbalance = scored.get("VoltageImbalancePct", pd.Series(0, index=scored.index)).fillna(0)

    scored["_EvidenceScore"] = (
        severity_rank * 1000
        + half_load * 200
        + probability * 100
        + voltage_deviation
        + voltage_imbalance
    )
    return scored


def unique_strongest_by_meter(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results.copy()

    scored = add_evidence_score(results)
    meter_column = "Meter Number"
    if meter_column not in scored.columns:
        scored[meter_column] = scored.index.astype(str)

    scored[meter_column] = scored[meter_column].astype(str).str.strip()
    scored.loc[scored[meter_column].eq("") | scored[meter_column].eq("nan"), meter_column] = (
        "ROW-" + scored.index.astype(str)
    )

    best_indices = scored.groupby(meter_column, dropna=False)["_EvidenceScore"].idxmax()
    return scored.loc[best_indices].sort_values("_EvidenceScore", ascending=False).drop(columns=["_EvidenceScore"])


def make_user_table(results: pd.DataFrame) -> pd.DataFrame:
    display = results.copy()
    if display.empty:
        return display

    display["الأولوية"] = display.apply(priority_label, axis=1)
    display["نوع المؤشر"] = display.apply(evidence_category, axis=1)
    display["المؤشر الرئيسي"] = display.apply(choose_indicator, axis=1)
    display["احتمال الفاقد"] = display["LossProbability"].map(probability_label)

    percent_columns = {
        "VoltageDeviationPct": "انحراف الجهد %",
        "VoltageImbalancePct": "عدم اتزان الجهد %",
        "CurrentImbalancePct": "عدم اتزان التيار %",
    }
    for source, target in percent_columns.items():
        if source in display.columns:
            display[target] = display[source].map(lambda value: "" if pd.isna(value) else f"{value:.2f}%")

    number_columns = {
        "MeanVoltage": "متوسط الجهد",
        "NominalVoltage": "الجهد الاسمي",
    }
    for source, target in number_columns.items():
        if source in display.columns:
            display[target] = display[source].map(lambda value: "" if pd.isna(value) else f"{value:.2f}")

    columns = [
        "Meter Number",
        "الأولوية",
        "نوع المؤشر",
        "احتمال الفاقد",
        "المؤشر الرئيسي",
        "متوسط الجهد",
        "الجهد الاسمي",
        "انحراف الجهد %",
        "عدم اتزان الجهد %",
        "عدم اتزان التيار %",
        "V1",
        "V2",
        "V3",
        "A1",
        "A2",
        "A3",
    ]
    columns = [column for column in columns if column in display.columns]

    return display[columns].rename(columns={"Meter Number": "رقم العداد/الآلة"})


def style_user_table(display: pd.DataFrame):
    def style_row(row: pd.Series) -> list[str]:
        marker_columns = {"الأولوية", "نوع المؤشر", "المؤشر الرئيسي", "احتمال الفاقد"}
        if row.get("الأولوية") == "حرجة":
            color = "background-color: #fff4e0; color: #4b2f08; font-weight: 700;"
        elif "جمبر" in str(row.get("نوع المؤشر", "")):
            color = "background-color: #eef7f5; color: #143f39; font-weight: 700;"
        else:
            color = "background-color: #f4f8fb; color: #18343a; font-weight: 700;"

        return [color if column in marker_columns else "" for column in row.index]

    return display.style.apply(style_row, axis=1)


def metric_card(label: str, value: str, note: str = "") -> str:
    note_html = f'<div class="metric-note">{escape(note)}</div>' if note else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{escape(label)}</div>
        <div class="metric-value">{escape(value)}</div>
        {note_html}
    </div>
    """


def build_user_excel(unique_results: pd.DataFrame, all_results: pd.DataFrame) -> bytes:
    output = BytesIO()
    user_table = make_user_table(unique_results)
    summary = pd.DataFrame(
        [
            {"Metric": "Total rows", "Value": len(all_results)},
            {"Metric": "Analyzed rows", "Value": int((all_results["AnalysisStatus"] == "Analyzed").sum())},
            {"Metric": "Unique confirmed meters", "Value": len(unique_results)},
            {"Metric": "Duplicate confirmed readings removed", "Value": int(all_results["FinalPotentialLoss"].sum()) - len(unique_results)},
        ]
    )

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        user_table.to_excel(writer, index=False, sheet_name="Confirmed Cases")
        summary.to_excel(writer, index=False, sheet_name="Summary")

    return output.getvalue()


def analyze(input_df: pd.DataFrame) -> pd.DataFrame:
    model_results = predict_loss(
        input_df,
        threshold=ANALYSIS_SETTINGS["threshold"],
        model=get_model(),
    )
    return engineering_review(
        model_results,
        enabled=ANALYSIS_SETTINGS["enabled"],
        treat_127v_as_normal=ANALYSIS_SETTINGS["treat_127v_as_normal"],
        treat_two_phase_line_line_as_normal=ANALYSIS_SETTINGS["treat_two_phase_line_line_as_normal"],
        confirmed_only=ANALYSIS_SETTINGS["confirmed_only"],
        require_vi_confirmed_for_final=ANALYSIS_SETTINGS["require_vi_confirmed_for_final"],
        vi_high_can_override_model=ANALYSIS_SETTINGS["vi_high_can_override_model"],
        vi_high_auto_confirm=ANALYSIS_SETTINGS["vi_high_auto_confirm"],
        include_half_load_jumper_suspect=ANALYSIS_SETTINGS["include_half_load_jumper_suspect"],
        half_load_jumper_min_probability_pct=ANALYSIS_SETTINGS["half_load_jumper_min_probability_pct"],
        use_current_imbalance_as_evidence=ANALYSIS_SETTINGS["use_current_imbalance_as_evidence"],
        voltage_tolerance_pct=ANALYSIS_SETTINGS["voltage_tolerance_pct"],
        voltage_imbalance_pct=ANALYSIS_SETTINGS["voltage_imbalance_pct"],
        current_imbalance_pct=ANALYSIS_SETTINGS["current_imbalance_pct"],
        two_phase_current_similarity_pct=ANALYSIS_SETTINGS["two_phase_current_similarity_pct"],
        inactive_phase_current_pct=ANALYSIS_SETTINGS["inactive_phase_current_pct"],
        strong_probability_pct=ANALYSIS_SETTINGS["strong_probability_pct"],
        committee_min_votes=ANALYSIS_SETTINGS["committee_min_votes"],
    )


st.markdown(
    """
    <div class="hero">
        <div class="hero-inner">
            <div>
                <div class="status-pill">تحليل عالي الثقة</div>
                <h1>تحليل الفاقد المحتمل</h1>
                <div class="app-subtitle">منصة ذكية لفرز قراءات الأحمال الكهربائية واستخراج العدادات الأعلى دلالة، مع اختيار أقوى قراءة لكل عداد.</div>
            </div>
            <div class="hero-panel">
                
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

upload_column, template_column = st.columns([1.45, 1], gap="large")
with upload_column:
    st.markdown(
        """
        <div class="workflow-card">
            <div class="workflow-title">ملف البيانات</div>
            <div class="workflow-note">صيغة Excel أو CSV بالأعمدة القياسية للجهد والتيار.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "ملف البيانات",
        type=["xlsx", "csv"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )
with template_column:
    st.markdown(
        """
        <div class="workflow-card">
            <div class="workflow-title">نموذج الإدخال</div>
            <div class="workflow-note">ملف جاهز للتعبئة بنفس أسماء الأعمدة المطلوبة.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.download_button(
        "تحميل نموذج البيانات",
        data=build_input_template(),
        file_name="meter_input_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

with st.expander("الأعمدة المطلوبة", expanded=False):
    st.dataframe(required_columns_table(), use_container_width=True, hide_index=True)

if uploaded_file is None:
    st.info("اختر ملف Excel أو CSV يحتوي أعمدة القراءات المطلوبة.")
    st.stop()

try:
    input_df = read_input_file(uploaded_file, uploaded_file.name)
    with st.spinner("جاري تحليل الملف..."):
        results_df = analyze(input_df)
except Exception as exc:
    st.error(f"تعذر تحليل الملف: {exc}")
    st.stop()

final_rows = results_df[results_df["FinalPotentialLoss"] == True].copy()
unique_final_rows = unique_strongest_by_meter(final_rows)
removed_duplicates = max(len(final_rows) - len(unique_final_rows), 0)
analyzed_count = int((results_df["AnalysisStatus"] == "Analyzed").sum())
invalid_count = int((results_df["AnalysisStatus"] != "Analyzed").sum())
vi_high_count = int((results_df["VIExpertSeverity"] == "High").sum()) if "VIExpertSeverity" in results_df.columns else 0
unique_vi_high_count = (
    int((unique_final_rows["VIExpertSeverity"] == "High").sum())
    if "VIExpertSeverity" in unique_final_rows.columns
    else 0
)
unique_jumper_count = (
    int(unique_final_rows["LikelyHalfLoadJumper"].sum())
    if "LikelyHalfLoadJumper" in unique_final_rows.columns
    else 0
)
highest_probability = (
    unique_final_rows["LossProbability"].max()
    if "LossProbability" in unique_final_rows.columns and not unique_final_rows.empty
    else pd.NA
)

st.markdown(
    '<div class="section-title"><h3>ملخص التشغيل</h3><span>تم تطبيق المعايير الفنية داخليًا قبل عرض القائمة النهائية</span></div>',
    unsafe_allow_html=True,
)
metric_columns = st.columns(4)
metric_columns[0].markdown(metric_card("إجمالي السجلات", f"{len(results_df):,}", "عدد القراءات المستلمة"), unsafe_allow_html=True)
metric_columns[1].markdown(metric_card("تم تحليلها", f"{analyzed_count:,}", "سجلات مكتملة وصالحة"), unsafe_allow_html=True)
metric_columns[2].markdown(metric_card("عدادات مؤكدة", f"{len(unique_final_rows):,}", "بعد إزالة التكرارات"), unsafe_allow_html=True)
metric_columns[3].markdown(metric_card("تكرارات مستبعدة", f"{removed_duplicates:,}", "احتفظنا بالأقوى دلالة"), unsafe_allow_html=True)

secondary_metrics = st.columns(2)
secondary_metrics[0].markdown(metric_card("حالات V/I مؤكدة", f"{vi_high_count:,}", "قرينة فولت/تيار مباشرة"), unsafe_allow_html=True)
secondary_metrics[1].markdown(metric_card("سجلات غير مكتملة", f"{invalid_count:,}", "لم تدخل في القرار النهائي"), unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="signal-strip">
        <div class="signal-item"><strong>{unique_vi_high_count:,}</strong>عدادات نهائية بقرينة V/I مؤكدة</div>
        <div class="signal-item"><strong>{unique_jumper_count:,}</strong>عدادات نهائية بشبهة نصف حمل/جمبر</div>
        <div class="signal-item"><strong>{probability_label(highest_probability)}</strong>أعلى احتمال فاقد في القائمة النهائية</div>
    </div>
    """,
    unsafe_allow_html=True,
)

left_column, right_column = st.columns([3, 1])
with left_column:
    st.markdown(
        '<div class="section-title"><h3>النتائج النهائية</h3><span>المؤشر الرئيسي مميز لسهولة ترتيب الزيارات الميدانية</span></div>',
        unsafe_allow_html=True,
    )
with right_column:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        "تصدير النتائج",
        data=build_user_excel(unique_final_rows, results_df),
        file_name=f"confirmed_loss_cases_{timestamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

display_df = make_user_table(unique_final_rows)

if display_df.empty:
    st.success("لا توجد حالات فاقد مؤكدة وفق المعايير الحالية.")
else:
    st.dataframe(
        style_user_table(display_df),
        use_container_width=True,
        hide_index=True,
        height=560,
        column_config={
            "رقم العداد/الآلة": st.column_config.TextColumn(width="medium"),
            "الأولوية": st.column_config.TextColumn(width="small"),
            "نوع المؤشر": st.column_config.TextColumn(width="medium"),
            "احتمال الفاقد": st.column_config.TextColumn(width="small"),
            "المؤشر الرئيسي": st.column_config.TextColumn(width="large"),
            "متوسط الجهد": st.column_config.TextColumn(width="small"),
            "الجهد الاسمي": st.column_config.TextColumn(width="small"),
            "انحراف الجهد %": st.column_config.TextColumn(width="small"),
            "عدم اتزان الجهد %": st.column_config.TextColumn(width="small"),
            "عدم اتزان التيار %": st.column_config.TextColumn(width="small"),
        },
    )

st.markdown(
    '<div class="app-footer">تطوير: مشهور العباس 2026 | 00966553339838</div>',
    unsafe_allow_html=True,
)
