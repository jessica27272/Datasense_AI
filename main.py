"""
DataSense AI — Autonomous Data Analyst Agent
Entry point: streamlit run main.py
"""

import os
import json
import tempfile

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataSense AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    [data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
    [data-testid="stSidebar"] * { color: #e6edf3 !important; }
    .hero-title {
        font-size: 2.4rem; font-weight: 600;
        background: linear-gradient(135deg, #58a6ff 0%, #79c0ff 50%, #a5d6ff 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; line-height: 1.2; margin-bottom: 0.3rem;
    }
    .hero-sub { color: #8b949e; font-size: 1rem; margin-bottom: 2rem; }
    .metric-card {
        background: #161b22; border: 1px solid #21262d;
        border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center;
    }
    .metric-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem; }
    .metric-value { font-size: 1.8rem; font-weight: 600; color: #58a6ff; font-family: 'DM Mono', monospace; }
    .metric-sub { font-size: 0.75rem; color: #6e7681; margin-top: 0.2rem; }
    .status-badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.78rem; font-weight: 500; }
    .badge-green { background: #1a4a2e; color: #3fb950; border: 1px solid #238636; }
    .badge-blue  { background: #1c2f4a; color: #58a6ff; border: 1px solid #1f6feb; }
    .badge-gray  { background: #21262d; color: #8b949e; border: 1px solid #30363d; }
    .chart-desc { font-size: 0.85rem; color: #8b949e; margin-top: 0.5rem; padding: 0 0.5rem; }
    .report-container {
        background: #161b22; border: 1px solid #21262d;
        border-radius: 12px; padding: 1.5rem 2rem; line-height: 1.8;
    }
    .stButton > button {
        background: #1f6feb; color: white; border: none; border-radius: 8px;
        font-weight: 500; padding: 0.55rem 1.4rem; transition: background 0.2s;
    }
    .stButton > button:hover { background: #388bfd; }
    hr { border-color: #21262d !important; margin: 1.5rem 0 !important; }
    </style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "df": None,
    "file_path": None,
    "analysis_result": None,
    "is_running": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── API Key check ─────────────────────────────────────────────────────────────
groq_key = os.getenv("GROQ_API_KEY", "")
cerebras_key = os.getenv("CEREBRAS_API_KEY", "")
api_ready = bool(groq_key or cerebras_key)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 DataSense AI")
    st.markdown("*Autonomous Data Analyst Agent*")
    st.markdown("---")

    if api_ready:
        provider = "Groq (Llama 3)" if groq_key else "Cerebras"
        st.markdown(f'<span class="status-badge badge-green">✓ {provider} Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge badge-gray">✗ No API Key Found</span>', unsafe_allow_html=True)
        st.markdown("Add to `.env`:")
        st.code("GROQ_API_KEY=your_key\n# get free at console.groq.com", language="bash")

    st.markdown("---")
    st.markdown("### 📂 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Drop CSV or Excel here",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        suffix = "." + uploaded_file.name.split(".")[-1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded_file.read())
        tmp.flush()
        st.session_state.file_path = tmp.name
        try:
            from tools import load_file
            st.session_state.df = load_file(tmp.name)
            st.markdown(f'<span class="status-badge badge-blue">📄 {uploaded_file.name}</span>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error loading file: {e}")

    st.markdown("---")
    st.markdown("### ⚙️ Options")
    user_question = st.text_area(
        "Custom question (optional)",
        placeholder="e.g. Which region has highest sales?",
        height=90,
    )
    st.markdown("---")
    st.markdown("""
        <div style="font-size:0.75rem; color:#6e7681; line-height:1.6;">
        <b style="color:#8b949e;">Powered by</b><br>
        🤖 Groq + Llama 3<br>
        🐼 Pandas + Matplotlib<br>
        🚀 100% Free Stack
        </div>
    """, unsafe_allow_html=True)


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">DataSense AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Upload your data → AI agents analyse, visualise, and explain it automatically.</div>', unsafe_allow_html=True)

if st.session_state.df is not None:
    df = st.session_state.df

    # Metric cards
    cols = st.columns(4)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    missing_pct = round(df.isnull().mean().mean() * 100, 1)

    metrics = [
        ("Rows", f"{df.shape[0]:,}", "records"),
        ("Columns", str(df.shape[1]), "features"),
        ("Numeric", str(len(num_cols)), "columns"),
        ("Missing", f"{missing_pct}%", "avg per col"),
    ]
    for col, (label, value, sub) in zip(cols, metrics):
        col.markdown(f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Preview", "📊 Column Info", "🔢 Quick Stats"])
    with tab1:
        st.dataframe(df.head(20), use_container_width=True, height=280)
    with tab2:
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Non-Null": df.notnull().sum().values,
            "Null %": (df.isnull().mean() * 100).round(1).astype(str) + "%",
            "Unique": df.nunique().values,
        })
        st.dataframe(info_df, use_container_width=True, hide_index=True)
    with tab3:
        if num_cols:
            st.dataframe(df[num_cols].describe().round(3), use_container_width=True)
        else:
            st.info("No numeric columns found.")

    st.markdown("---")

    # Run button
    if not st.session_state.is_running:
        run_col, _ = st.columns([1, 3])
        with run_col:
            if st.button("🚀 Run AI Analysis", use_container_width=True):
                if not api_ready:
                    st.error("❌ Add GROQ_API_KEY or CEREBRAS_API_KEY to .env first!")
                else:
                    st.session_state.is_running = True
                    st.session_state.analysis_result = None
                    st.rerun()

    # Running state
    if st.session_state.is_running:
        st.markdown("### 🤖 AI Agents at Work...")
        with st.spinner("Agents are analysing your data... This takes 1-2 minutes."):
            from crew_runner import run_analysis
            result = run_analysis(
                file_path=st.session_state.file_path,
                user_question=user_question,
            )
        st.session_state.analysis_result = result
        st.session_state.is_running = False
        st.rerun()

    # Results
    if st.session_state.analysis_result is not None:
        result = st.session_state.analysis_result

        if result.get("error"):
            st.error(f"❌ Analysis failed: {result['error']}")
        else:
            st.markdown('<span class="status-badge badge-green">✅ Analysis Complete</span>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            res_tab1, res_tab2, res_tab3 = st.tabs(["📝 Analysis Report", "📊 Charts", "💾 Export"])

            with res_tab1:
                report = result.get("analysis_report", "No report generated.")
                st.markdown(f'<div class="report-container">{report}</div>', unsafe_allow_html=True)

            with res_tab2:
                charts = result.get("charts", [])
                if not charts:
                    st.info("No charts were generated.")
                else:
                    for i in range(0, len(charts), 2):
                        chart_cols = st.columns(2)
                        for j, col in enumerate(chart_cols):
                            if i + j < len(charts):
                                chart = charts[i + j]
                                path = chart.get("path", "")
                                if os.path.exists(path):
                                    with col:
                                        st.image(path, use_column_width=True)
                                        st.markdown(f'<div class="chart-desc"><b>{chart.get("title","")}</b><br>{chart.get("description","")}</div>', unsafe_allow_html=True)

            with res_tab3:
                st.markdown("### 💾 Download Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "📄 Download Report (.md)",
                        data=result.get("analysis_report", "").encode("utf-8"),
                        file_name="datasense_report.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
                with col2:
                    st.download_button(
                        "📊 Download Chart Info (.json)",
                        data=json.dumps(result.get("charts", []), indent=2).encode("utf-8"),
                        file_name="datasense_charts.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                for chart in result.get("charts", []):
                    path = chart.get("path", "")
                    if os.path.exists(path):
                        with open(path, "rb") as f:
                            st.download_button(
                                f"🖼️ {chart.get('title', os.path.basename(path))}",
                                data=f.read(),
                                file_name=os.path.basename(path),
                                mime="image/png",
                                use_container_width=True,
                            )

else:
    # Empty state
    st.markdown("<br>", unsafe_allow_html=True)
    empty_col1, empty_col2, empty_col3 = st.columns([1, 2, 1])
    with empty_col2:
        st.markdown("""
            <div style="text-align:center; padding:3rem 2rem; background:#161b22;
                        border:2px dashed #21262d; border-radius:16px;">
                <div style="font-size:3rem; margin-bottom:1rem;">🧬</div>
                <div style="font-size:1.2rem; font-weight:600; color:#e6edf3; margin-bottom:0.5rem;">
                    Upload a dataset to begin
                </div>
                <div style="color:#8b949e; font-size:0.9rem; line-height:1.6;">
                    Drop any CSV or Excel file in the sidebar.<br>
                    Two AI agents will automatically analyse and visualise your data.
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### ⚡ How It Works")
    hw_cols = st.columns(3)
    steps = [
        ("1️⃣ Upload", "Drop any CSV or Excel file in the sidebar."),
        ("2️⃣ AI Analyses", "Analyst Agent finds trends, correlations and anomalies using Llama 3."),
        ("3️⃣ Visualise & Export", "Visualizer Agent creates charts. Download report + charts."),
    ]
    for col, (title, desc) in zip(hw_cols, steps):
        col.markdown(f"""<div class="metric-card" style="text-align:left;">
            <div style="font-size:1.2rem; margin-bottom:0.5rem;">{title}</div>
            <div style="font-size:0.85rem; color:#8b949e; line-height:1.6;">{desc}</div>
        </div>""", unsafe_allow_html=True)