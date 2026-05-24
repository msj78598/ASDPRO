from __future__ import annotations

from datetime import datetime
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
            background: #f5f7f6;
            color: #1e2420;
        }
        [data-testid="stSidebar"] {
            display: none;
        }
        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        h1 {
            font-size: 2.15rem;
            margin-bottom: 0.2rem;
            color: #15211e;
            font-weight: 750;
        }
        .app-subtitle {
            color: #59645f;
            margin-bottom: 0.8rem;
            font-size: 1rem;
            line-height: 1.8;
        }
        .hero {
            background: #ffffff;
            border: 1px solid #dbe3dd;
            border-radius: 8px;
            padding: 1.05rem 1.15rem;
            box-shadow: 0 1px 3px rgba(30, 36, 32, 0.05);
            margin-bottom: 0.9rem;
        }
        .workflow-card {
            background: #ffffff;
            border: 1px solid #dbe3dd;
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            min-height: 4.2rem;
            margin-bottom: 0.55rem;
        }
        .workflow-title {
            color: #31433d;
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
            background: #e8f0ec;
            color: #28574f;
            border: 1px solid #cdded6;
            border-radius: 999px;
            padding: 0.25rem 0.65rem;
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 0.55rem;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #dde4de;
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            box-shadow: 0 1px 2px rgba(30, 36, 32, 0.05);
        }
        [data-testid="stMetricLabel"] {
            color: #53605a;
        }
        [data-testid="stMetricValue"] {
            color: #16362f;
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
        [data-testid="stDataFrame"] {
            border: 1px solid #dde4de;
            border-radius: 8px;
            overflow: hidden;
        }
        .stAlert {
            border-radius: 8px;
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
        <div class="status-pill">تحليل عالي الثقة</div>
        <h1>تحليل الفاقد المحتمل</h1>
        <div class="app-subtitle">قائمة نهائية مختصرة بالعدادات الأعلى دلالة، مع اختيار أقوى قراءة لكل عداد.</div>
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

metric_columns = st.columns(4)
metric_columns[0].metric("إجمالي السجلات", f"{len(results_df):,}")
metric_columns[1].metric("تم تحليلها", f"{analyzed_count:,}")
metric_columns[2].metric("عدادات مؤكدة", f"{len(unique_final_rows):,}")
metric_columns[3].metric("تكرارات مستبعدة", f"{removed_duplicates:,}")

secondary_metrics = st.columns(2)
secondary_metrics[0].metric("حالات V/I مؤكدة", f"{vi_high_count:,}")
secondary_metrics[1].metric("سجلات غير مكتملة", f"{invalid_count:,}")

st.divider()

left_column, right_column = st.columns([3, 1])
with left_column:
    st.subheader("النتائج النهائية")
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
        display_df,
        use_container_width=True,
        hide_index=True,
        height=560,

st.markdown("---")
st.markdown(
    "<p style='font-size:12px;color:#777;'>👨‍💻 تطوير: مشهور العباس 2026 | 00966553339838</p>",
    unsafe_allow_html=True,
    )
