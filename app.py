from __future__ import annotations

from datetime import datetime
from html import escape
from io import BytesIO

import pandas as pd
import streamlit as st

from predict_loss import engineering_review, load_model, predict_loss, read_input_file


st.set_page_config(
    page_title="تحليل الفاقد المحتمل",
    page_icon="📊",
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
    "current_diversion_imbalance_pct": 60.0,
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
        @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800;900&display=swap');
        :root {
            --se-blue: #0078FC;
            --se-blue-700: #0A57C2;
            --se-navy: #143C90;
            --se-navy-900: #0C2A66;
            --se-teal: #23C2C8;
            --se-cyan: #7FD8E8;
            --se-cyan-soft: #A8E4F0;
            --se-bg: #EEF4FB;
            --se-ink: #15294A;
            --se-muted: #5B6B82;
            --se-border: #D8E3F2;
            --se-card: #FFFFFF;
        }
        .stApp {
            direction: rtl;
            text-align: right;
            background: var(--se-bg);
            color: var(--se-ink);
        }
        html, body, [class*="css"] {
            font-family: "Tajawal", "Segoe UI", Tahoma, Arial, sans-serif;
        }
        [data-testid="stHeader"] {
            display: none;
        }
        [data-testid="stToolbar"] {
            display: none;
        }
        [data-testid="stSidebar"] {
            display: none;
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1220px;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        h1 {
            font-size: 2.25rem;
            margin: 0 0 0.35rem 0;
            color: #ffffff;
            font-weight: 800;
        }
        .app-subtitle {
            color: #dceaff;
            margin: 0;
            font-size: 1.02rem;
            line-height: 1.8;
        }
        .topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            padding: 0.95rem 1.2rem;
            margin-bottom: 1rem;
            background: linear-gradient(90deg, var(--se-navy) 0%, var(--se-blue) 100%);
            border-radius: 12px;
            color: #eef6ff;
            box-shadow: 0 10px 32px rgba(20, 60, 144, 0.22);
        }
        .topbar h1 {
            margin: 0;
            font-size: 1.55rem;
            font-weight: 900;
            letter-spacing: 0.02em;
        }
        .topbar .brand-note {
            color: #cfe2ff;
            font-size: 0.95rem;
            line-height: 1.5;
            max-width: 42rem;
        }
        .hero {
            position: relative;
            overflow: hidden;
            color: #eef6ff;
            display: flex;
            align-items: center;
            min-height: 170px;
            background:
                url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20width='440'%20height='210'%20viewBox='0%200%20440%20210'%3E%3Cpolyline%20points='10,160%2075,125%20140,140%20205,80%20270,105%20335,45%20430,72'%20fill='none'%20stroke='%23ffffff'%20stroke-width='3'%20stroke-opacity='0.15'/%3E%3Cg%20fill='%23ffffff'%20fill-opacity='0.20'%3E%3Ccircle%20cx='205'%20cy='80'%20r='5'/%3E%3Ccircle%20cx='335'%20cy='45'%20r='5'/%3E%3C/g%3E%3C/svg%3E") left -10px center / 420px auto no-repeat,
                radial-gradient(circle at 16% 28%, rgba(127, 216, 232, 0.30), transparent 32%),
                linear-gradient(135deg, var(--se-navy-900) 0%, var(--se-navy) 48%, var(--se-blue) 100%);
            border: 1px solid rgba(35, 194, 200, 0.30);
            border-radius: 14px;
            padding: 1.7rem 1.95rem;
            box-shadow: 0 16px 34px rgba(12, 42, 102, 0.24);
            margin-bottom: 1.1rem;
        }
        .hero h1 {
            color: #ffffff;
            font-size: 2.35rem;
            font-weight: 800;
            margin: 0.15rem 0 0.5rem 0;
            text-shadow: 0 2px 16px rgba(2, 18, 52, 0.38);
        }
        .metric-card, .workflow-card, .signal-item {
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }
        .metric-card:hover, .workflow-card:hover, .signal-item:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 30px rgba(20, 60, 144, 0.15);
        }
        .hero::before {
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(90deg, transparent 0 18%, rgba(255,255,255,0.06) 18% 18.12%, transparent 18.12% 100%),
                linear-gradient(180deg, transparent 0 68%, rgba(255,255,255,0.06) 68% 68.16%, transparent 68.16% 100%);
            opacity: 0.6;
            pointer-events: none;
        }
        .hero::after {
            content: "";
            position: absolute;
            top: 0;
            right: -30%;
            width: 32%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(127, 216, 232, 0.22), transparent);
            animation: scan-line 8s ease-in-out infinite;
            pointer-events: none;
        }
        @keyframes scan-line {
            0%, 30% { right: -32%; opacity: 0; }
            45%, 70% { opacity: 1; }
            100% { right: 102%; opacity: 0; }
        }
        .hero-inner {
            position: relative;
            display: flex;
            flex-direction: column;
            gap: 0.9rem;
        }
        .hero-panel {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.7rem;
            border: 1px solid rgba(255, 255, 255, 0.14);
            background: rgba(255, 255, 255, 0.09);
            border-radius: 8px;
            padding: 0.85rem;
            backdrop-filter: blur(6px);
        }
        .module-panel {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.95rem;
            margin-bottom: 1rem;
        }
        .module-card {
            background: var(--se-card);
            border: 1px solid var(--se-border);
            border-top: 3px solid var(--se-blue);
            border-radius: 12px;
            padding: 1.1rem 1.15rem;
            box-shadow: 0 10px 24px rgba(20, 60, 144, 0.08);
            min-height: 6.2rem;
        }
        .module-title {
            color: var(--se-navy);
            font-weight: 800;
            font-size: 1rem;
            margin: 0 0 0.45rem 0;
        }
        .module-desc {
            color: var(--se-muted);
            font-size: 0.92rem;
            line-height: 1.6;
            margin: 0;
        }
        .panel-row {
            display: flex;
            justify-content: space-between;
            gap: 0.8rem;
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 8px;
            padding: 0.65rem 0.75rem;
            color: #dceaff;
            font-size: 0.88rem;
            background: rgba(10, 30, 70, 0.22);
        }
        .panel-row span {
            text-align: right;
        }
        .panel-row:last-child {
            border-bottom: 1px solid rgba(255, 255, 255, 0.16);
        }
        .panel-row strong {
            color: #ffffff;
            font-weight: 800;
        }
        .workflow-card {
            background: var(--se-card);
            border: 1px solid var(--se-border);
            border-right: 3px solid var(--se-teal);
            border-radius: 8px;
            padding: 0.9rem 1rem;
            min-height: 5rem;
            margin: 0 0 0.55rem 0;
            box-shadow: 0 4px 14px rgba(20, 60, 144, 0.06);
        }
        .workflow-title {
            color: var(--se-navy);
            font-weight: 800;
            font-size: 0.95rem;
            margin-bottom: 0.45rem;
        }
        .workflow-note {
            color: var(--se-muted);
            font-size: 0.88rem;
            line-height: 1.7;
            margin-bottom: 0.65rem;
        }
        .panel-head {
            color: var(--se-navy);
            font-weight: 800;
            font-size: 1.02rem;
            margin-bottom: 0.2rem;
        }
        .panel-sub {
            color: var(--se-muted);
            font-size: 0.88rem;
            line-height: 1.7;
            margin-bottom: 0.7rem;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.panel-head) {
            background: var(--se-card);
            border: 1px solid var(--se-border) !important;
            border-radius: 12px;
            box-shadow: 0 6px 18px rgba(20, 60, 144, 0.06);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.panel-head):hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 30px rgba(20, 60, 144, 0.13);
        }
        .status-pill {
            display: inline-block;
            background: rgba(255,255,255,0.14);
            color: #ffffff;
            border: 1px solid rgba(255,255,255,0.26);
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
            background: var(--se-teal);
            box-shadow: 0 0 0 4px rgba(35, 194, 200, 0.20);
        }
        .metric-card {
            background: var(--se-card);
            border: 1px solid var(--se-border);
            border-radius: 8px;
            padding: 1rem 1.05rem;
            box-shadow: 0 4px 14px rgba(20, 60, 144, 0.06);
            min-height: 7.1rem;
            position: relative;
            overflow: hidden;
        }
        .metric-card::before {
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 4px;
            background: linear-gradient(90deg, var(--se-blue), var(--se-teal));
        }
        .metric-label {
            color: var(--se-muted);
            font-size: 0.88rem;
            margin-bottom: 0.55rem;
        }
        .metric-value {
            color: var(--se-navy);
            font-weight: 800;
            font-size: 2rem;
            line-height: 1.1;
        }
        .metric-note {
            color: #8493a8;
            font-size: 0.78rem;
            margin-top: 0.5rem;
        }
        .stButton > button {
            border-radius: 8px;
            border: 1px solid var(--se-blue);
            background: var(--se-blue);
            color: white;
            min-height: 2.8rem;
            font-weight: 750;
            transition: background 0.15s ease, transform 0.05s ease;
        }
        .stButton > button:hover {
            border-color: var(--se-blue-700);
            background: var(--se-blue-700);
            color: white;
        }
        .stButton > button:active {
            transform: translateY(1px);
        }
        [data-testid="stProgress"] > div > div > div > div {
            background-image: linear-gradient(90deg, var(--se-blue), var(--se-blue-700));
        }
        .stDownloadButton > button {
            border-radius: 8px;
            border: 1px solid var(--se-blue);
            background: var(--se-blue);
            color: white;
            min-height: 2.7rem;
            font-weight: 700;
        }
        .stDownloadButton > button:hover {
            border-color: var(--se-blue-700);
            background: var(--se-blue-700);
            color: white;
        }
        [data-testid="stFileUploader"] {
            background: #f6f9fe;
            border: 1px dashed #aac3e8;
            border-radius: 8px;
            padding: 0.7rem;
            margin-top: 0.1rem;
        }
        [data-testid="stFileUploaderDropzone"] {
            min-height: 5.2rem;
            border-radius: 8px;
            background: #eaf1fb;
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
            border: 1px solid var(--se-border);
            border-radius: 8px;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--se-border);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 14px rgba(20, 60, 144, 0.06);
        }
        .stAlert {
            border-radius: 8px;
        }
        .stButton > button {
            border-radius: 8px;
            min-height: 2.7rem;
            font-weight: 700;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, var(--se-navy), var(--se-blue));
            border: none;
            box-shadow: 0 8px 20px rgba(20, 60, 144, 0.22);
        }
        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(90deg, var(--se-navy-900), var(--se-blue-700));
        }
        [data-testid="stProgress"] {
            margin: 0.5rem 0 0.2rem 0;
        }
        [data-testid="stProgress"] [role="progressbar"] {
            height: 13px;
            border-radius: 999px;
            overflow: hidden;
            background-color: #d8e3f2;
        }
        [data-testid="stProgress"] [role="progressbar"] > div {
            background: linear-gradient(90deg, var(--se-blue), var(--se-teal)) !important;
            border-radius: 999px;
        }
        .app-footer {
            color: var(--se-muted);
            border-top: 1px solid var(--se-border);
            font-size: 0.85rem;
            margin-top: 1.4rem;
            padding-top: 0.9rem;
            text-align: center;
        }
        .section-title {
            display: flex;
            justify-content: flex-start;
            align-items: center;
            gap: 0.75rem;
            flex-wrap: wrap;
            margin: 0.4rem 0 0.7rem 0;
            min-height: 2.85rem;
            background: linear-gradient(90deg, var(--se-navy) 0%, var(--se-blue) 100%);
            border-radius: 10px;
            padding: 0.5rem 1.05rem;
            box-shadow: 0 6px 18px rgba(20, 60, 144, 0.16);
        }
        .section-title h3 {
            margin: 0;
            color: #ffffff;
            font-size: 1.1rem;
            font-weight: 800;
        }
        .section-title span {
            color: #cfe2ff;
            font-size: 0.85rem;
            padding-right: 0.75rem;
            border-right: 1px solid rgba(255, 255, 255, 0.28);
        }
        .results-shell {
            background: var(--se-card);
            border: 1px solid var(--se-border);
            border-radius: 8px;
            padding: 0.95rem;
            box-shadow: 0 8px 24px rgba(20, 60, 144, 0.07);
        }
        .signal-strip {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.7rem;
            margin: 0.35rem 0 0.85rem 0;
        }
        .signal-item {
            background: #f4f9ff;
            border: 1px solid var(--se-border);
            border-top: 3px solid var(--se-teal);
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            color: var(--se-muted);
            font-size: 0.86rem;
        }
        .signal-item strong {
            display: block;
            color: var(--se-navy);
            font-size: 0.96rem;
            margin-bottom: 0.25rem;
        }
        [data-testid="stExpander"] {
            background: rgba(255,255,255,0.9);
            border: 1px solid var(--se-border);
            border-radius: 8px;
        }
        @media (max-width: 860px) {
            .hero-panel {
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


def current_diversion_suspects(results: pd.DataFrame, min_share: float = 0.7) -> pd.DataFrame:
    """عدّادات يغلب على قراءاتها نمط اشتباه تحويل التيار (جمبر)، قراءة واحدة لكل عداد."""
    if "CurrentDiversionSuspect" not in results.columns or results.empty:
        return results.iloc[0:0].copy()

    data = results.copy()
    meter_column = "Meter Number"
    if meter_column not in data.columns:
        data[meter_column] = data.index.astype(str)
    data[meter_column] = data[meter_column].astype(str).str.strip()

    share = data.groupby(meter_column)["CurrentDiversionSuspect"].transform("mean")
    flagged = data[(share >= min_share) & data["CurrentDiversionSuspect"]].copy()
    if flagged.empty:
        return flagged

    strongest = flagged.groupby(meter_column)["CurrentImbalancePct"].idxmax()
    return flagged.loc[strongest].sort_values("CurrentImbalancePct", ascending=False)


def make_diversion_table(rows: pd.DataFrame) -> pd.DataFrame:
    display = rows.copy()
    if display.empty:
        return display

    table = pd.DataFrame(index=display.index)
    table["رقم العداد/الآلة"] = (
        display["Meter Number"].astype(str) if "Meter Number" in display.columns else display.index.astype(str)
    )
    table["نوع الاشتباه"] = "تحويل تيار (جمبر)"
    table["التوصية"] = "تأكيد ميداني بالكلامب على الكابلات"
    if "CurrentImbalancePct" in display.columns:
        table["عدم اتزان التيار %"] = display["CurrentImbalancePct"].map(
            lambda value: "" if pd.isna(value) else f"{value:.1f}%"
        )
    for column in ["A1", "A2", "A3", "V1", "V2", "V3"]:
        if column in display.columns:
            table[column] = display[column]
    return table.reset_index(drop=True)


def df_to_excel_bytes(frame: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()


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
                <div class="app-subtitle">تحليل قراءات الجهد والتيار لرصد حالات الفاقد المحتمل بدقة عالية.</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title"><h3>رفع بيانات القراءات</h3><span>ارفع ملف القراءات أو حمّل النموذج الجاهز للتعبئة</span></div>',
    unsafe_allow_html=True,
)

template_column, upload_column = st.columns(2, gap="large")
with upload_column:
    with st.container(border=True):
        st.markdown(
            """
            <div class="panel-head">ملف البيانات</div>
            <div class="panel-sub">صيغة Excel أو CSV بالأعمدة القياسية للجهد والتيار.</div>
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
    with st.container(border=True):
        st.markdown(
            """
            <div class="panel-head">نموذج الإدخال</div>
            <div class="panel-sub">ملف جاهز للتعبئة بنفس أسماء الأعمدة المطلوبة.</div>
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
    st.session_state.pop("results_df", None)
    st.session_state.pop("analyzed_file", None)
    st.stop()

# إذا تغيّر الملف، أزل نتائج التشغيل السابق
if st.session_state.get("analyzed_file") != uploaded_file.name:
    st.session_state.pop("results_df", None)

start_col, hint_col = st.columns([1, 2.2])
with start_col:
    start_clicked = st.button("▶  بدء التحليل", type="primary", use_container_width=True)
with hint_col:
    st.caption("اضغط لبدء معالجة الملف. قد تستغرق الملفات الكبيرة بضع لحظات، وسيظهر شريط التقدم أدناه.")

if start_clicked:
    progress = st.progress(0, text="جاري تجهيز الملف…")
    try:
        input_df = read_input_file(uploaded_file, uploaded_file.name)
        row_count = len(input_df)
        progress.progress(8, text=f"تم تحميل البيانات ({row_count:,} سجل) — جاري التحليل…")

        # تحليل النموذج على دفعات حتى يتحرّك شريط التقدم بسلاسة مع الملفات الكبيرة
        chunk_size = max(2000, (row_count // 20) + 1)
        total_chunks = max(1, (row_count + chunk_size - 1) // chunk_size)
        parts = []
        start_index = 0
        done_chunks = 0
        while start_index < row_count:
            chunk = input_df.iloc[start_index : start_index + chunk_size]
            parts.append(
                predict_loss(chunk, threshold=ANALYSIS_SETTINGS["threshold"], model=get_model())
            )
            done_chunks += 1
            progress.progress(
                min(80, 8 + int(72 * done_chunks / total_chunks)),
                text=f"تحليل القراءات… ({done_chunks}/{total_chunks})",
            )
            start_index += chunk_size

        model_results = (
            pd.concat(parts)
            if parts
            else predict_loss(input_df, threshold=ANALYSIS_SETTINGS["threshold"], model=get_model())
        )

        progress.progress(88, text="المراجعة الفنية ولجنة الخبراء…")
        review_kwargs = {key: value for key, value in ANALYSIS_SETTINGS.items() if key != "threshold"}
        results_df = engineering_review(model_results, **review_kwargs)

        st.session_state["results_df"] = results_df
        st.session_state["analyzed_file"] = uploaded_file.name
        progress.progress(100, text="اكتمل التحليل ✓")
    except Exception as exc:
        progress.empty()
        st.error(f"تعذر تحليل الملف: {exc}")
        st.stop()

if "results_df" not in st.session_state:
    st.info("الملف جاهز. اضغط «بدء التحليل» لعرض النتائج.")
    st.stop()

results_df = st.session_state["results_df"]

final_rows = results_df[results_df["FinalPotentialLoss"] == True].copy()
unique_final_rows = unique_strongest_by_meter(final_rows)
removed_duplicates = max(len(final_rows) - len(unique_final_rows), 0)

diversion_rows = current_diversion_suspects(results_df)
if not diversion_rows.empty and not unique_final_rows.empty and "Meter Number" in unique_final_rows.columns:
    confirmed_meters = set(unique_final_rows["Meter Number"].astype(str).str.strip())
    diversion_rows = diversion_rows[
        ~diversion_rows["Meter Number"].astype(str).str.strip().isin(confirmed_meters)
    ]
diversion_count = len(diversion_rows)
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
metric_row_one = st.columns(3)
metric_row_one[0].markdown(metric_card("إجمالي السجلات", f"{len(results_df):,}", "عدد القراءات المستلمة"), unsafe_allow_html=True)
metric_row_one[1].markdown(metric_card("تم تحليلها", f"{analyzed_count:,}", "سجلات مكتملة وصالحة"), unsafe_allow_html=True)
metric_row_one[2].markdown(metric_card("سجلات غير مكتملة", f"{invalid_count:,}", "لم تدخل في القرار النهائي"), unsafe_allow_html=True)

metric_row_two = st.columns(3)
metric_row_two[0].markdown(metric_card("عدادات مؤكدة", f"{len(unique_final_rows):,}", "بعد إزالة التكرارات"), unsafe_allow_html=True)
metric_row_two[1].markdown(metric_card("حالات V/I مؤكدة", f"{vi_high_count:,}", "قرينة فولت/تيار مباشرة"), unsafe_allow_html=True)
metric_row_two[2].markdown(metric_card("تكرارات مستبعدة", f"{removed_duplicates:,}", "احتفظنا بالأقوى دلالة"), unsafe_allow_html=True)

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

if diversion_count:
    st.markdown(
        f"""
        <div class="signal-strip">
            <div class="signal-item" style="border-top-color:#E8A23A;background:#fff7ec;color:#7a4a09;">
                <strong style="color:#9a5a06;">{diversion_count:,}</strong>
                عدّادات باشتباه تحويل تيار (جمبر) — جهود سليمة مع عدم اتزان تيار شديد، تحتاج تأكيد ميداني
            </div>
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

if diversion_count:
    st.markdown(
        '<div class="section-title"><h3>اشتباه تحويل تيار (جمبر)</h3><span>جهود سليمة مع عدم اتزان تيار شديد — يحتاج تأكيد ميداني بالكلامب على الكابلات</span></div>',
        unsafe_allow_html=True,
    )
    st.warning(
        f"تم رصد {diversion_count:,} عدّاد بنمط اشتباه تحويل تيار. هذه ليست فاقداً مؤكداً، بل أولوية تفتيش ميداني: "
        "قِس التيار الفعلي بالكلامب على الكابلات وقارنه بما يسجّله العدّاد."
    )
    diversion_table = make_diversion_table(diversion_rows)
    diversion_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        "تصدير اشتباهات تحويل التيار",
        data=df_to_excel_bytes(diversion_table, sheet_name="Current Diversion"),
        file_name=f"current_diversion_suspects_{diversion_timestamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.dataframe(
        diversion_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "رقم العداد/الآلة": st.column_config.TextColumn(width="medium"),
            "نوع الاشتباه": st.column_config.TextColumn(width="small"),
            "التوصية": st.column_config.TextColumn(width="medium"),
            "عدم اتزان التيار %": st.column_config.TextColumn(width="small"),
        },
    )

st.markdown(
    '<div class="app-footer">تطوير: مشهور العباس 2026 | 00966553339838</div>',
    unsafe_allow_html=True,
)
