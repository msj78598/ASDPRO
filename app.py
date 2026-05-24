from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from predict_loss import (
    build_committee_summary,
    build_excel,
    engineering_review,
    load_model,
    predict_loss,
    read_input_file,
)


st.set_page_config(
    page_title="تحليل الفاقد المحتمل",
    layout="wide",
)


st.markdown(
    """
    <style>
        .stApp {
            direction: rtl;
            text-align: right;
            background: #f6f7f4;
            color: #1e2420;
        }
        [data-testid="stSidebar"] {
            direction: rtl;
            text-align: right;
            background: #20322e;
        }
        [data-testid="stSidebar"] * {
            color: #f7f4ec;
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1320px;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        h1 {
            font-size: 2rem;
            margin-bottom: 0.25rem;
            color: #15211e;
        }
        .app-subtitle {
            color: #59645f;
            margin-bottom: 1.2rem;
            font-size: 1rem;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #dde4de;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            box-shadow: 0 1px 2px rgba(30, 36, 32, 0.05);
        }
        [data-testid="stMetricLabel"] {
            color: #53605a;
        }
        [data-testid="stMetricValue"] {
            color: #16362f;
        }
        .stButton > button,
        .stDownloadButton > button {
            border-radius: 8px;
            border: 1px solid #2e6f64;
            background: #2e6f64;
            color: white;
            min-height: 2.6rem;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: #244f49;
            background: #244f49;
            color: white;
        }
        [data-testid="stFileUploader"] {
            background: #ffffff;
            border: 1px solid #dde4de;
            border-radius: 8px;
            padding: 0.85rem;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid #dde4de;
            border-radius: 8px;
            overflow: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def probability_label(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.1%}"


def make_display_table(results: pd.DataFrame) -> pd.DataFrame:
    display = results.copy()
    if "LossProbability" in display.columns:
        display["LossProbability"] = display["LossProbability"].map(probability_label)

    percent_columns = ["VoltageDeviationPct", "VoltageImbalancePct", "CurrentImbalancePct"]
    for column in percent_columns:
        if column in display.columns:
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.2f}%")

    number_columns = ["MeanVoltage", "NominalVoltage"]
    for column in number_columns:
        if column in display.columns:
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.2f}")

    columns = [
        column
        for column in [
            "Meter Number",
            "LossProbability",
            "PotentialLoss",
            "FinalPotentialLoss",
            "CommitteeLossVotes",
            "CommitteeNormalVetoes",
            "CommitteeDecision",
            "VIExpertSeverity",
            "VIExpertStatus",
            "VIExpertReasons",
            "LikelyHalfLoadJumper",
            "EngineeringDecision",
            "EngineeringReason",
            "MeanVoltage",
            "NominalVoltage",
            "VoltageDeviationPct",
            "VoltageImbalancePct",
            "CurrentImbalancePct",
            "LikelyTwoPhaseLineLine",
            "AnalysisStatus",
            "V1",
            "V2",
            "V3",
            "A1",
            "A2",
            "A3",
        ]
        if column in display.columns
    ]

    display = display[columns]
    return display.rename(
        columns={
            "Meter Number": "رقم العداد/الآلة",
            "LossProbability": "احتمال الفاقد",
            "PotentialLoss": "اشتباه النموذج",
            "FinalPotentialLoss": "القرار النهائي",
            "CommitteeLossVotes": "أصوات اللجنة",
            "CommitteeNormalVetoes": "اعتراضات فنية",
            "CommitteeDecision": "قرار لجنة الخبراء",
            "VIExpertSeverity": "تقييم خبير V/I",
            "VIExpertStatus": "حكم خبير V/I",
            "VIExpertReasons": "سبب خبير V/I",
            "LikelyHalfLoadJumper": "شبهة نصف حمل/جمبر",
            "EngineeringDecision": "قرار اللجنة الفنية",
            "EngineeringReason": "سبب القرار",
            "MeanVoltage": "متوسط الجهد",
            "NominalVoltage": "الجهد الاسمي",
            "VoltageDeviationPct": "انحراف الجهد %",
            "VoltageImbalancePct": "عدم اتزان الجهد %",
            "CurrentImbalancePct": "عدم اتزان التيار %",
            "LikelyTwoPhaseLineLine": "نمط حار-حار",
            "AnalysisStatus": "حالة التحليل",
        }
    )


@st.cache_resource
def get_model():
    return load_model()


st.title("تحليل الفاقد المحتمل")
st.markdown(
    "<div class='app-subtitle'>رفع بيانات القراءات، تشغيل النموذج المدرب، ثم تمرير الحالات المشتبه بها على مراجعة فنية قابلة للضبط.</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("الإعدادات")
    threshold = st.slider(
        "حد اعتبار الفاقد",
        min_value=0.05,
        max_value=0.95,
        value=0.95,
        step=0.05,
        format="%.2f",
    )

    st.subheader("اللجنة الفنية")
    enable_review = st.toggle("تفعيل المراجعة الفنية", value=True)
    confirmed_only = st.toggle("إظهار الحالات المؤكدة فقط", value=True)
    require_vi_confirmed = st.toggle("اعتماد قرينة V/I مؤكدة فقط", value=True)
    vi_high_override = st.toggle("رفع حالات V/I المؤكدة حتى لو لم يحللها النموذج", value=True)
    vi_high_auto_confirm = st.toggle("اعتماد V/I High مباشرة", value=True)
    include_half_load_jumper = st.toggle("إدراج شبهة نصف حمل/جمبر", value=True)
    half_load_jumper_probability = st.slider("أقل احتمال لشبهة نصف حمل/جمبر %", 95.0, 100.0, 99.5, 0.5)
    treat_127v_as_normal = st.toggle("اعتبار 127V جهدًا تشغيليًا طبيعيًا", value=True)
    treat_two_phase_line_line = st.toggle("استبعاد نمط حار-حار على فازين", value=True)
    two_phase_current_similarity = st.slider("أقصى فرق بين تياري حار-حار %", 5, 30, 15, 5)
    inactive_phase_current = st.slider("حد الفازة الثالثة شبه المعدومة %", 5, 40, 20, 5)
    use_current_imbalance = st.toggle("استخدام عدم اتزان التيار كدليل مستقل", value=False)
    strong_probability = st.slider("احتمال قوي يبقى كفاقد", 50, 95, 95, 5)
    committee_min_votes = st.slider("أقل أصوات لاعتماد الفاقد", 5, 10, 9, 1)
    voltage_tolerance = st.slider("سماحية انحراف الجهد %", 3, 30, 20, 1)
    voltage_imbalance = st.slider("حد عدم اتزان الجهد %", 1.0, 30.0, 20.0, 0.5)
    current_imbalance = st.slider("حد عدم اتزان التيار %", 10, 100, 30, 5)

    view_mode = st.radio(
        "العرض",
        ["النتيجة النهائية", "اشتباه النموذج قبل اللجنة", "الحالات المستبعدة فنيا", "كل النتائج"],
    )

uploaded_file = st.file_uploader(
    "ملف البيانات",
    type=["xlsx", "csv"],
    accept_multiple_files=False,
)

if uploaded_file is None:
    st.info("اختر ملف Excel أو CSV يحتوي أعمدة القراءات.")
    st.stop()

try:
    input_df = read_input_file(uploaded_file, uploaded_file.name)
    model_results_df = predict_loss(input_df, threshold=threshold, model=get_model())
    results_df = engineering_review(
        model_results_df,
        enabled=enable_review,
        treat_127v_as_normal=treat_127v_as_normal,
        treat_two_phase_line_line_as_normal=treat_two_phase_line_line,
        confirmed_only=confirmed_only,
        require_vi_confirmed_for_final=require_vi_confirmed,
        vi_high_can_override_model=vi_high_override,
        vi_high_auto_confirm=vi_high_auto_confirm,
        include_half_load_jumper_suspect=include_half_load_jumper,
        half_load_jumper_min_probability_pct=half_load_jumper_probability,
        use_current_imbalance_as_evidence=use_current_imbalance,
        voltage_tolerance_pct=voltage_tolerance,
        voltage_imbalance_pct=voltage_imbalance,
        current_imbalance_pct=current_imbalance,
        two_phase_current_similarity_pct=two_phase_current_similarity,
        inactive_phase_current_pct=inactive_phase_current,
        strong_probability_pct=strong_probability,
        committee_min_votes=committee_min_votes,
    )
except Exception as exc:
    st.error(f"تعذر تحليل الملف: {exc}")
    st.stop()

analyzed_count = int((results_df["AnalysisStatus"] == "Analyzed").sum())
model_suspected_df = results_df[results_df["PotentialLoss"] == True].sort_values(
    "LossProbability",
    ascending=False,
)
final_suspected_df = results_df[results_df["FinalPotentialLoss"] == True].sort_values(
    "LossProbability",
    ascending=False,
)
filtered_df = results_df[
    (results_df["PotentialLoss"] == True) & (results_df["FinalPotentialLoss"] == False)
].sort_values("LossProbability", ascending=False)
invalid_count = int((results_df["AnalysisStatus"] != "Analyzed").sum())

metric_columns = st.columns(4)
metric_columns[0].metric("إجمالي السجلات", f"{len(results_df):,}")
metric_columns[1].metric("تم تحليلها", f"{analyzed_count:,}")
metric_columns[2].metric("اشتباه النموذج", f"{len(model_suspected_df):,}")
metric_columns[3].metric("النتيجة النهائية", f"{len(final_suspected_df):,}")

vi_high_count = int((results_df["VIExpertSeverity"] == "High").sum()) if "VIExpertSeverity" in results_df.columns else 0
secondary_metrics = st.columns(3)
secondary_metrics[0].metric("مستبعدة فنيا", f"{len(filtered_df):,}")
secondary_metrics[1].metric("حالات V/I مؤكدة", f"{vi_high_count:,}")
secondary_metrics[2].metric("غير مكتملة", f"{invalid_count:,}")

committee_summary = build_committee_summary(results_df)
with st.expander("محضر وتوصيات لجنة الخبراء", expanded=True):
    st.dataframe(committee_summary, use_container_width=True, hide_index=True)

st.divider()

left_column, right_column = st.columns([3, 1])
with left_column:
    st.subheader("النتائج")
with right_column:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        "تصدير Excel",
        data=build_excel(results_df, threshold),
        file_name=f"loss_analysis_{timestamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

if view_mode == "كل النتائج":
    table_source = results_df
elif view_mode == "اشتباه النموذج قبل اللجنة":
    table_source = model_suspected_df
elif view_mode == "الحالات المستبعدة فنيا":
    table_source = filtered_df
else:
    table_source = final_suspected_df

display_df = make_display_table(table_source)

if display_df.empty:
    st.success("لا توجد سجلات في هذا العرض عند الإعدادات الحالية.")
else:
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=520,
    )
