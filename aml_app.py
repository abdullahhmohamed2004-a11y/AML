"""
AML Detection Pipeline — Streamlit UI
Author: Abdullah Hasan Mohamed | ID: 20221104
AI Subject — College Project
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import io
import time

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AML Detection Pipeline",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0a0e1a;
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1226 0%, #111827 100%);
    border-right: 1px solid #1e293b;
}

/* Headers */
h1, h2, h3 { font-family: 'Sora', sans-serif; font-weight: 700; }
h1 { color: #38bdf8; letter-spacing: -0.5px; }
h2 { color: #94a3b8; font-size: 1.1rem; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; }
h3 { color: #e2e8f0; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #38bdf8; }
.metric-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px; font-family: 'Space Mono', monospace; }
.metric-value { font-size: 2.2rem; font-weight: 700; color: #38bdf8; font-family: 'Space Mono', monospace; }
.metric-sub { font-size: 0.8rem; color: #475569; margin-top: 4px; }

/* Phase badges */
.phase-badge {
    display: inline-block;
    background: linear-gradient(90deg, #0ea5e9, #6366f1);
    color: white;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 999px;
    letter-spacing: 1px;
    font-family: 'Space Mono', monospace;
    margin-bottom: 8px;
}

/* Alert boxes */
.alert-fraud {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 14px 18px;
    color: #fca5a5;
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
}
.alert-safe {
    background: rgba(34, 197, 94, 0.08);
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-left: 4px solid #22c55e;
    border-radius: 8px;
    padding: 14px 18px;
    color: #86efac;
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
}

/* Divider */
.section-divider {
    border: none;
    border-top: 1px solid #1e293b;
    margin: 24px 0;
}

/* Code mono font */
.mono { font-family: 'Space Mono', monospace; }

/* Sticker header */
.hero-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(56,189,248,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title { font-size: 2rem; font-weight: 700; color: #f1f5f9; margin: 0; }
.hero-sub { color: #64748b; font-size: 0.9rem; margin-top: 6px; font-family: 'Space Mono', monospace; }
.hero-tag {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.4);
    color: #a5b4fc;
    font-size: 0.7rem;
    padding: 3px 10px;
    border-radius: 999px;
    margin-top: 12px;
    font-family: 'Space Mono', monospace;
    letter-spacing: 1px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #0ea5e9, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Inputs */
.stNumberInput input, .stSelectbox select {
    background: #111827 !important;
    border: 1px solid #334155 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* Progress */
.stProgress > div > div { background: linear-gradient(90deg, #0ea5e9, #6366f1) !important; }

/* Tabs */
.stTabs [data-baseweb="tab"] { font-family: 'Sora', sans-serif; font-weight: 600; color: #64748b; }
.stTabs [aria-selected="true"] { color: #38bdf8 !important; }

/* Tables */
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* Matplotlib dark */
plt_bg = "#111827"
</style>
""", unsafe_allow_html=True)


# ── Matplotlib dark theme ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#111827",
    "axes.facecolor":    "#111827",
    "axes.edgecolor":    "#334155",
    "axes.labelcolor":   "#94a3b8",
    "xtick.color":       "#64748b",
    "ytick.color":       "#64748b",
    "text.color":        "#e2e8f0",
    "grid.color":        "#1e293b",
    "grid.alpha":        0.8,
    "axes.grid":         True,
    "axes.titlecolor":   "#f1f5f9",
    "axes.titlesize":    12,
    "font.family":       "monospace",
})

ACCENT   = "#38bdf8"
ACCENT2  = "#6366f1"
RED      = "#f87171"
GREEN    = "#4ade80"
ORANGE   = "#fb923c"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 24px;'>
        <div style='font-size:2.5rem;'>🛡️</div>
        <div style='font-weight:700; font-size:1.1rem; color:#38bdf8; margin-top:8px;'>AML Detector</div>
        <div style='font-size:0.72rem; color:#475569; font-family:Space Mono,monospace; margin-top:4px;'>
            Abdullah Hasan Mohamed<br>ID: 20221104
        </div>
    </div>
    <hr style='border-color:#1e293b; margin:0 0 20px;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠 Overview", "📊 EDA & Data", "🔬 PCA Analysis",
         "🔵 Clustering", "⚡ XGBoost Model", "🔄 Batch Learning",
         "🧪 Transaction Scorer"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#1e293b; margin:20px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.7rem; color:#334155; font-family:Space Mono,monospace; line-height:1.8;'>
    PIPELINE PHASES<br>
    ✅ Phase 1 — Preprocessing<br>
    ✅ Phase 2 — PCA<br>
    ✅ Phase 3 — K-Means<br>
    ✅ Phase 4 — XGBoost<br>
    ✅ Phase 5 — Batch Learning
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: load data  (handles files of any size)
# ═══════════════════════════════════════════════════════════════════════════════
REQUIRED_COLS = [
    "step", "type", "amount",
    "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest",
    "isFraud", "isFlaggedFraud",
]

@st.cache_data(show_spinner=False)
def load_data_from_path(path: str, sample_frac: float = 1.0) -> pd.DataFrame:
    """Load CSV from a local file path, optionally sampling rows."""
    # Count total rows cheaply (no full read)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        total_rows = sum(1 for _ in f) - 1  # subtract header

    if sample_frac < 1.0:
        # Stratified-style sample: keep all fraud rows + a sample of legit rows
        # We do two cheap passes using skiprows
        import random
        rng = random.Random(SEED)
        keep_rows = set()
        # First pass: find fraud row indices
        chunk_iter = pd.read_csv(path, chunksize=100_000,
                                  usecols=["isFraud"], low_memory=True)
        offset = 0
        fraud_idx = []
        for chunk in chunk_iter:
            for local_i, val in enumerate(chunk["isFraud"].values):
                if val == 1:
                    fraud_idx.append(offset + local_i)
            offset += len(chunk)
        # Sample legit rows
        all_idx = list(range(total_rows))
        legit_idx = [i for i in all_idx if i not in set(fraud_idx)]
        n_legit_keep = int(len(legit_idx) * sample_frac)
        sampled_legit = set(rng.sample(legit_idx, min(n_legit_keep, len(legit_idx))))
        keep_rows = sampled_legit | set(fraud_idx)
        # Read only kept rows
        skip = [i + 1 for i in range(total_rows) if i not in keep_rows]  # +1 for header
        df = pd.read_csv(path, skiprows=skip, low_memory=False)
    else:
        df = pd.read_csv(path, low_memory=False)

    return df


@st.cache_data(show_spinner=False)
def load_data_from_upload(uploaded_file, sample_frac: float = 1.0) -> pd.DataFrame:
    """Load CSV from a Streamlit UploadedFile object."""
    df = pd.read_csv(uploaded_file, low_memory=False)
    if sample_frac < 1.0:
        fraud = df[df["isFraud"] == 1]
        legit = df[df["isFraud"] == 0].sample(frac=sample_frac, random_state=SEED)
        df = pd.concat([fraud, legit]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.preprocessing import LabelEncoder
    df = df.copy()
    df.drop(columns=["isFlaggedFraud", "nameOrig", "nameDest"], errors="ignore", inplace=True)
    le = LabelEncoder()
    df["type_encoded"] = le.fit_transform(df["type"])
    df["orig_balance_diff"]             = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["dest_balance_diff"]             = df["newbalanceDest"] - df["oldbalanceDest"]
    df["amount_to_orig_balance_ratio"]  = np.where(df["oldbalanceOrg"] > 0,
                                                    df["amount"] / df["oldbalanceOrg"], 0)
    df["orig_balance_zero_after"]       = (df["newbalanceOrig"] == 0).astype(int)
    df["dest_balance_zero_before"]      = (df["oldbalanceDest"] == 0).astype(int)
    return df


CONTINUOUS_FEATURES = [
    "step", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "orig_balance_diff", "dest_balance_diff", "amount_to_orig_balance_ratio",
]
BINARY_FEATURES = ["type_encoded", "orig_balance_zero_after", "dest_balance_zero_before"]
TARGET = "isFraud"
SEED = 42


def fig_to_st(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🛡️ Anti-Money Laundering Detection</div>
        <div class="hero-sub">End-to-End Machine Learning Pipeline · PaySim Synthetic Dataset</div>
        <div class="hero-tag">AI SUBJECT · COLLEGE PROJECT · 2022</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    phases = [
        ("01", "Preprocessing", "Feature Engineering & Scaling"),
        ("02", "PCA", "Dimensionality Reduction"),
        ("03", "K-Means", "Behavioral Segmentation"),
        ("04", "XGBoost", "Risk Scoring"),
        ("05", "Batch", "Continuous Learning"),
    ]
    for col, (num, title, sub) in zip([col1, col2, col3, col4, col5], phases):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">PHASE {num}</div>
                <div style="font-size:1rem; font-weight:700; color:#f1f5f9; margin:4px 0;">{title}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    st.markdown("## 📂 LOAD DATASET")
    st.markdown("""
    <p style='color:#94a3b8; font-size:0.9rem;'>
    The PaySim dataset is ~500 MB – 1.5 GB. Choose the method that suits your setup.
    Dataset available on <a href='https://www.kaggle.com/datasets/ealaxi/paysim1'
    target='_blank' style='color:#38bdf8;'>Kaggle</a>.
    </p>
    """, unsafe_allow_html=True)

    load_mode = st.radio(
        "Loading method",
        ["📁 Local file path (recommended for large files)",
         "⬆️  File uploader (≤ 2 GB, needs .streamlit/config.toml)"],
        horizontal=True,
    )

    col_load, col_frac = st.columns([3, 1])
    with col_frac:
        st.markdown("<br>", unsafe_allow_html=True)
        sample_pct = st.slider(
            "Sample % of legit rows",
            min_value=10, max_value=100, value=100, step=10,
            help="100% = load everything. Lower values speed up training on slow machines. ALL fraud rows are always kept."
        )
        sample_frac = sample_pct / 100.0

    df_raw = None
    with col_load:
        if "Local file path" in load_mode:
            st.markdown("<br>", unsafe_allow_html=True)
            csv_path = st.text_input(
                "Full path to your paysim CSV",
                placeholder=r"e.g.  C:\Users\Abdullah\Downloads\paysim.csv  or  /home/user/paysim.csv",
            )
            if st.button("🚀 Load from path") and csv_path:
                import os
                if not os.path.isfile(csv_path):
                    st.error(f"❌ File not found: `{csv_path}`")
                else:
                    size_mb = os.path.getsize(csv_path) / 1024 / 1024
                    with st.spinner(f"Loading {size_mb:.0f} MB file… this may take 30–60 s for large files."):
                        df_raw = load_data_from_path(csv_path, sample_frac=sample_frac)
                    st.session_state["df_raw"] = df_raw
                    st.session_state["csv_path"] = csv_path
        else:
            uploaded = st.file_uploader(
                "Drop your paysim CSV here  (max 2 GB — see .streamlit/config.toml)",
                type=["csv"],
            )
            if uploaded:
                size_mb = uploaded.size / 1024 / 1024
                with st.spinner(f"Reading {size_mb:.0f} MB… hang tight."):
                    df_raw = load_data_from_upload(uploaded, sample_frac=sample_frac)
                st.session_state["df_raw"] = df_raw

    # ── show stats once loaded ─────────────────────────────────────────────────
    if df_raw is None and "df_raw" in st.session_state:
        df_raw = st.session_state["df_raw"]

    if df_raw is not None:
        r, c = df_raw.shape
        fraud_count = int(df_raw["isFraud"].sum())
        fraud_rate  = df_raw["isFraud"].mean() * 100

        m1, m2, m3, m4 = st.columns(4)
        for col, label, val, sub in [
            (m1, "ROWS LOADED",        f"{r:,}",           f"{sample_pct}% sample" if sample_frac < 1 else "full dataset"),
            (m2, "FEATURES",           f"{c}",              "raw columns"),
            (m3, "FRAUD CASES",        f"{fraud_count:,}",  f"{fraud_rate:.3f}% rate"),
            (m4, "LEGITIMATE",         f"{r - fraud_count:,}", "clean transactions"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{val}</div>
                    <div class="metric-sub">{sub}</div>
                </div>
                """, unsafe_allow_html=True)

        st.success("✅ Dataset loaded! Navigate the sidebar to explore each pipeline phase.")
        st.dataframe(df_raw.head(8), use_container_width=True, height=280)
    else:
        st.markdown("""
        <div style='background:rgba(56,189,248,0.07); border:1px solid rgba(56,189,248,0.2);
                    border-radius:10px; padding:16px 20px; color:#94a3b8; font-size:0.88rem;'>
        💡 <strong>Tip for large files:</strong> Use the <em>Local file path</em> option — it reads the CSV
        directly from disk without the browser upload overhead, so there's no size limit at all.
        You can also reduce the <em>Sample %</em> slider to 30–50% to speed up training
        while keeping all fraud rows intact.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("## 🏗️ PIPELINE ARCHITECTURE")
    st.markdown("""
    | Phase | Component | Method | Output |
    |---|---|---|---|
    | 1 | Data Preprocessing | Label Encoding + StandardScaler | Clean feature matrix |
    | 2 | Dimensionality Reduction | PCA (90% variance) | Orthogonal components |
    | 3 | Behavioral Segmentation | K-Means + Silhouette | `behavioral_segment` feature |
    | 4 | Risk Scoring | XGBoost (scale_pos_weight) | Fraud probability [0–1] |
    | 5 | Continuous Learning | Warm-start incremental training | Updated model |
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA & Data":
    st.markdown('<div class="phase-badge">PHASE 1 — PREPROCESSING</div>', unsafe_allow_html=True)
    st.title("Exploratory Data Analysis")

    if "df_raw" not in st.session_state:
        st.warning("⬅️  Please upload the dataset on the Overview page first.")
        st.stop()

    df = st.session_state["df_raw"].copy()

    # Type distribution + fraud by type
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Transaction Type Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = df["type"].value_counts()
        bars = ax.bar(counts.index, counts.values,
                      color=[ACCENT, ACCENT2, "#22d3ee", ORANGE, GREEN],
                      edgecolor="#0a0e1a", linewidth=0.8)
        ax.set_xlabel("Transaction Type")
        ax.set_ylabel("Count")
        ax.set_title("Volume by Type")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000,
                    f"{bar.get_height():,.0f}", ha="center", va="bottom", fontsize=8, color="#94a3b8")
        fig_to_st(fig)

    with col_b:
        st.markdown("### Fraud Rate (%) by Transaction Type")
        fig, ax = plt.subplots(figsize=(6, 4))
        fraud_by_type = df.groupby("type")["isFraud"].mean() * 100
        colors = [RED if v > 0 else "#334155" for v in fraud_by_type.values]
        bars = ax.bar(fraud_by_type.index, fraud_by_type.values,
                      color=colors, edgecolor="#0a0e1a", linewidth=0.8)
        ax.set_xlabel("Transaction Type")
        ax.set_ylabel("Fraud Rate (%)")
        ax.set_title("Fraud Concentration")
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=8, color=RED)
        fig_to_st(fig)

    st.markdown("""
    <div class="alert-fraud">
    ⚠️  <strong>Key Insight:</strong> Fraud exclusively occurs in <strong>TRANSFER</strong> and <strong>CASH_OUT</strong>
    transaction types. All other types (CASH_IN, DEBIT, PAYMENT) carry zero fraud.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # Amount distribution
    st.markdown("### Transaction Amount Distribution (log scale)")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    legit = df[df["isFraud"] == 0]["amount"]
    fraud = df[df["isFraud"] == 1]["amount"]
    axes[0].hist(np.log1p(legit.sample(min(50000, len(legit)), random_state=SEED)),
                 bins=60, color=ACCENT, alpha=0.7, label="Legitimate")
    axes[0].hist(np.log1p(fraud), bins=60, color=RED, alpha=0.8, label="Fraud")
    axes[0].set_xlabel("log(1 + Amount)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Amount Distribution (log)")
    axes[0].legend()

    axes[1].boxplot([np.log1p(legit.sample(min(10000, len(legit)), random_state=SEED)),
                     np.log1p(fraud)],
                    labels=["Legitimate", "Fraud"],
                    patch_artist=True,
                    boxprops=dict(facecolor=ACCENT2, color=ACCENT),
                    medianprops=dict(color=RED, linewidth=2))
    axes[1].set_ylabel("log(1 + Amount)")
    axes[1].set_title("Amount Boxplot by Class")
    fig_to_st(fig)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # Class imbalance
    st.markdown("### 🎯 Class Imbalance")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">FRAUD RATE</div>
            <div class="metric-value">{df["isFraud"].mean()*100:.3f}%</div>
            <div class="metric-sub">Severe imbalance → use scale_pos_weight</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 3))
        vals = df["isFraud"].value_counts()
        wedges, texts, autotexts = ax.pie(
            vals, labels=["Legitimate", "Fraud"],
            autopct="%1.3f%%", colors=[ACCENT, RED],
            startangle=90, wedgeprops=dict(edgecolor="#0a0e1a", linewidth=1.5),
            textprops=dict(color="#94a3b8", fontsize=9)
        )
        autotexts[0].set_color("#94a3b8")
        autotexts[1].set_color(RED)
        ax.set_title("Class Distribution")
        fig_to_st(fig)

    # Engineered features preview
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Engineered Features Preview")
    df_eng = engineer_features(df)
    new_cols = ["orig_balance_diff", "dest_balance_diff",
                "amount_to_orig_balance_ratio", "orig_balance_zero_after", "dest_balance_zero_before"]
    st.dataframe(df_eng[new_cols + ["isFraud"]].head(20), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PCA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 PCA Analysis":
    st.markdown('<div class="phase-badge">PHASE 2 — DIMENSIONALITY REDUCTION</div>', unsafe_allow_html=True)
    st.title("Principal Component Analysis")

    if "df_raw" not in st.session_state:
        st.warning("⬅️  Please upload the dataset on the Overview page first.")
        st.stop()

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    df = engineer_features(st.session_state["df_raw"])

    st.markdown("**Fitting PCA on the full feature matrix...**")

    with st.spinner("Computing PCA (this takes a moment on large datasets)..."):
        scaler_full = StandardScaler()
        X_cont = scaler_full.fit_transform(df[CONTINUOUS_FEATURES])
        X_unsupervised = np.hstack([X_cont, df[BINARY_FEATURES].values])

        pca_full = PCA(random_state=SEED)
        pca_full.fit(X_unsupervised)

        explained_var = pca_full.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        n_components_90 = int(np.argmax(cumulative_var >= 0.90)) + 1

    col1, col2, col3 = st.columns(3)
    for col, label, val, sub in [
        (col1, "TOTAL FEATURES",     f"{X_unsupervised.shape[1]}", "input dimensions"),
        (col2, "COMPONENTS @ 90%",   f"{n_components_90}",         "retained components"),
        (col3, "VARIANCE RETAINED",  f"{cumulative_var[n_components_90-1]*100:.1f}%", "information kept"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Scree
    axes[0].bar(range(1, len(explained_var)+1), explained_var*100,
                color=ACCENT, edgecolor="#0a0e1a", linewidth=0.5, alpha=0.85)
    axes[0].axvline(n_components_90, color=RED, linestyle="--", linewidth=1.5,
                    label=f"n = {n_components_90}")
    axes[0].set_title("Scree Plot")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance (%)")
    axes[0].legend()

    # Cumulative
    axes[1].plot(range(1, len(cumulative_var)+1), cumulative_var*100,
                 color=ACCENT, marker="o", markersize=4, linewidth=2)
    axes[1].axhline(90, color=GREEN, linestyle="--", linewidth=1.5, label="90% threshold")
    axes[1].axvline(n_components_90, color=RED, linestyle="--", linewidth=1.5,
                    label=f"n = {n_components_90}")
    axes[1].fill_between(range(1, n_components_90+1),
                         cumulative_var[:n_components_90]*100, alpha=0.1, color=ACCENT)
    axes[1].set_title("Cumulative Explained Variance")
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Variance (%)")
    axes[1].legend()

    fig_to_st(fig)

    # Loadings heatmap
    st.markdown("### PCA Loadings Heatmap")
    pca_final = PCA(n_components=n_components_90, random_state=SEED)
    pca_final.fit(X_unsupervised)
    feature_names = CONTINUOUS_FEATURES + BINARY_FEATURES
    loadings = pd.DataFrame(
        pca_final.components_.T,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(n_components_90)]
    )
    fig, ax = plt.subplots(figsize=(max(8, n_components_90), 6))
    sns.heatmap(loadings, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 8})
    ax.set_title("PCA Component Loadings")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    fig_to_st(fig)

    st.markdown(f"""
    <div class="alert-safe">
    ✅  <strong>Justification:</strong> We retain <strong>{n_components_90} principal components</strong>
    because they explain <strong>{cumulative_var[n_components_90-1]*100:.2f}%</strong> of total variance,
    satisfying the 90% threshold while discarding noise dimensions.
    </div>
    """, unsafe_allow_html=True)

    st.session_state["pca_model"]     = pca_final
    st.session_state["scaler_full"]   = scaler_full
    st.session_state["X_unsupervised"] = X_unsupervised
    st.session_state["df_engineered"] = df


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔵 Clustering":
    st.markdown('<div class="phase-badge">PHASE 3 — BEHAVIORAL SEGMENTATION</div>', unsafe_allow_html=True)
    st.title("K-Means Customer Clustering")

    if "df_raw" not in st.session_state:
        st.warning("⬅️  Please upload the dataset on the Overview page first.")
        st.stop()

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    with st.spinner("Preparing data for clustering..."):
        df = engineer_features(st.session_state["df_raw"])
        scaler_full = StandardScaler()
        X_cont = scaler_full.fit_transform(df[CONTINUOUS_FEATURES])
        X_unsup = np.hstack([X_cont, df[BINARY_FEATURES].values])

        pca_full = PCA(random_state=SEED)
        pca_full.fit(X_unsup)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        n_comp = int(np.argmax(cumvar >= 0.90)) + 1
        pca_model = PCA(n_components=n_comp, random_state=SEED)
        X_pca = pca_model.fit_transform(X_unsup)
        X_pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_comp)])

    SAMPLE_SIZE = min(30_000, len(X_pca_df))
    X_sample = X_pca_df.sample(n=SAMPLE_SIZE, random_state=SEED)

    k_range = range(2, 9)
    with st.spinner("Running elbow + silhouette analysis (k = 2..8)..."):
        wcss, sil = [], []
        prog = st.progress(0)
        for i, k in enumerate(k_range):
            km = KMeans(n_clusters=k, init="k-means++", random_state=SEED, n_init=10)
            labels = km.fit_predict(X_sample)
            wcss.append(km.inertia_)
            sil.append(silhouette_score(X_sample, labels, sample_size=5000, random_state=SEED))
            prog.progress((i+1) / len(k_range))

    best_k = list(k_range)[int(np.argmax(sil))]

    col1, col2 = st.columns(2)
    for col, label, val, sub in [
        (col1, "OPTIMAL K",        f"{best_k}",        "by silhouette score"),
        (col2, "BEST SILHOUETTE",  f"{max(sil):.4f}",  "cluster quality metric"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(list(k_range), wcss, marker="o", color=ACCENT, linewidth=2)
    axes[0].set_title("Elbow Method (WCSS)")
    axes[0].set_xlabel("k (Number of Clusters)")
    axes[0].set_ylabel("Within-Cluster Sum of Squares")

    axes[1].plot(list(k_range), sil, marker="o", color=ORANGE, linewidth=2)
    axes[1].axvline(best_k, linestyle="--", color=GREEN, label=f"Best k = {best_k}")
    axes[1].set_title("Silhouette Score by k")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].legend()
    fig_to_st(fig)

    # Final clustering
    st.markdown("### Cluster Visualization (PC1 vs PC2)")
    km_final = KMeans(n_clusters=best_k, init="k-means++", random_state=SEED, n_init=10)
    df["behavioral_segment"] = km_final.fit_predict(X_pca_df)

    plot_idx = np.random.choice(len(X_pca_df), size=min(15000, len(X_pca_df)), replace=False)
    pca_plot = X_pca_df.iloc[plot_idx]
    seg_plot  = df["behavioral_segment"].iloc[plot_idx]

    cmap = plt.cm.get_cmap("tab10", best_k)
    fig, ax = plt.subplots(figsize=(10, 5))
    for k in range(best_k):
        mask = seg_plot == k
        ax.scatter(pca_plot.loc[mask, "PC1"], pca_plot.loc[mask, "PC2"],
                   s=4, alpha=0.35, color=cmap(k), label=f"Cluster {k}")
    ax.set_title(f"K-Means Clusters (k={best_k}) in PCA Space")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(markerscale=3, fontsize=8)
    fig_to_st(fig)

    # Cluster profile
    st.markdown("### Cluster Profile")
    profile = df.groupby("behavioral_segment").agg(
        Transactions=("isFraud", "count"),
        Fraud_Rate=("isFraud", "mean"),
        Avg_Amount=("amount", "mean"),
        Avg_Orig_Balance=("oldbalanceOrg", "mean"),
    ).round(4)
    profile["Fraud_Rate"] = (profile["Fraud_Rate"] * 100).map("{:.3f}%".format)
    profile["Avg_Amount"] = profile["Avg_Amount"].map("${:,.2f}".format)
    profile["Avg_Orig_Balance"] = profile["Avg_Orig_Balance"].map("${:,.2f}".format)
    st.dataframe(profile, use_container_width=True)

    st.session_state["kmeans_model"]    = km_final
    st.session_state["pca_model"]       = pca_model
    st.session_state["scaler_full"]     = scaler_full
    st.session_state["df_engineered"]   = df
    st.session_state["best_k"]          = best_k


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: XGBOOST
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚡ XGBoost Model":
    st.markdown('<div class="phase-badge">PHASE 4 — PREDICTIVE RISK SCORING</div>', unsafe_allow_html=True)
    st.title("XGBoost Fraud Classifier")

    if "df_raw" not in st.session_state:
        st.warning("⬅️  Please upload the dataset on the Overview page first.")
        st.stop()

    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (precision_score, recall_score, f1_score,
                                  confusion_matrix, classification_report)

    st.markdown("### ⚙️ Hyperparameters")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1: max_depth    = st.slider("Max Depth",       3, 10, 6)
    with col_p2: lr           = st.slider("Learning Rate",   0.01, 0.3, 0.05, 0.01)
    with col_p3: n_rounds     = st.slider("Boost Rounds",    50, 500, 300, 50)
    with col_p4: threshold    = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05)

    if st.button("🚀 Train XGBoost Model"):
        with st.spinner("Building full pipeline and training..."):
            df = engineer_features(st.session_state["df_raw"])
            scaler_full = StandardScaler()
            X_cont = scaler_full.fit_transform(df[CONTINUOUS_FEATURES])
            X_unsup = np.hstack([X_cont, df[BINARY_FEATURES].values])

            pca_full_tmp = PCA(random_state=SEED)
            pca_full_tmp.fit(X_unsup)
            cumvar = np.cumsum(pca_full_tmp.explained_variance_ratio_)
            n_comp = int(np.argmax(cumvar >= 0.90)) + 1
            pca_model = PCA(n_components=n_comp, random_state=SEED)
            X_pca = pca_model.fit_transform(X_unsup)

            km = KMeans(n_clusters=4, init="k-means++", random_state=SEED, n_init=10)
            df["behavioral_segment"] = km.fit_predict(X_pca)

            MODEL_FEATURES = CONTINUOUS_FEATURES + BINARY_FEATURES + ["behavioral_segment"]
            X = df[MODEL_FEATURES].values
            y = df[TARGET].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=SEED, stratify=y)

            scaler_train = StandardScaler()
            cont_idx = list(range(len(CONTINUOUS_FEATURES)))
            X_train_c = scaler_train.fit_transform(X_train[:, cont_idx])
            X_test_c  = scaler_train.transform(X_test[:, cont_idx])
            X_train_f = np.hstack([X_train_c, X_train[:, len(CONTINUOUS_FEATURES):]])
            X_test_f  = np.hstack([X_test_c,  X_test[:, len(CONTINUOUS_FEATURES):]])

            neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
            spw = neg / pos

            dtrain = xgb.DMatrix(X_train_f, label=y_train)
            dtest  = xgb.DMatrix(X_test_f,  label=y_test)

            params = {
                "objective": "binary:logistic",
                "eval_metric": ["logloss", "aucpr"],
                "max_depth": max_depth,
                "learning_rate": lr,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 5,
                "scale_pos_weight": spw,
                "seed": SEED,
                "tree_method": "hist",
            }
            evals_result = {}
            model = xgb.train(
                params, dtrain, num_boost_round=n_rounds,
                evals=[(dtrain, "train"), (dtest, "eval")],
                early_stopping_rounds=20,
                evals_result=evals_result,
                verbose_eval=False,
            )

        y_prob = model.predict(dtest)
        y_pred = (y_prob >= threshold).astype(int)

        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)

        st.markdown("### 📈 Model Performance")
        m1, m2, m3 = st.columns(3)
        for col, label, val, color in [
            (m1, "PRECISION", f"{prec:.4f}", ACCENT),
            (m2, "RECALL",    f"{rec:.4f}",  GREEN),
            (m3, "F1-SCORE",  f"{f1:.4f}",   ORANGE),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="color:{color};">{val}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Pred: Legit", "Pred: Fraud"],
                        yticklabels=["True: Legit", "True: Fraud"],
                        annot_kws={"size": 13, "weight": "bold"})
            ax.set_title("Confusion Matrix")
            fig_to_st(fig)

        with col_b:
            st.markdown("### Learning Curves")
            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            for key, ax, title, color in [
                ("logloss", axes[0], "Log Loss",  RED),
                ("aucpr",   axes[1], "AUC-PR",    GREEN),
            ]:
                if key in evals_result.get("train", {}):
                    ax.plot(evals_result["train"][key], label="Train",
                            color=ACCENT, linewidth=1.5)
                    ax.plot(evals_result["eval"][key], label="Val",
                            color=color, linewidth=1.5, linestyle="--")
                    ax.set_title(title)
                    ax.set_xlabel("Round")
                    ax.legend(fontsize=8)
            fig_to_st(fig)

        # Feature importance
        st.markdown("### Feature Importance (Gain)")
        feature_names = CONTINUOUS_FEATURES + BINARY_FEATURES + ["behavioral_segment"]
        importance_dict = model.get_score(importance_type="gain")
        imp_named = {}
        for k, v in importance_dict.items():
            if k.startswith("f") and k[1:].isdigit():
                idx = int(k[1:])
                if idx < len(feature_names):
                    imp_named[feature_names[idx]] = v
        if imp_named:
            imp_df = pd.DataFrame.from_dict(imp_named, orient="index",
                                             columns=["Importance"]).sort_values("Importance")
            fig, ax = plt.subplots(figsize=(9, max(4, len(imp_df)*0.4)))
            colors = [RED if f in ["orig_balance_diff", "amount_to_orig_balance_ratio",
                                    "behavioral_segment"] else ACCENT for f in imp_df.index]
            ax.barh(imp_df.index, imp_df["Importance"], color=colors, edgecolor="#0a0e1a")
            ax.set_title("XGBoost Feature Importance (Gain)")
            ax.set_xlabel("Gain")
            fig_to_st(fig)

        st.session_state["xgb_model"]    = model
        st.session_state["scaler_train"] = scaler_train
        st.session_state["kmeans_model"] = km
        st.session_state["pca_model"]    = pca_model
        st.session_state["scaler_full"]  = scaler_full
        st.session_state["threshold"]    = threshold
        st.session_state["df_engineered"] = df
        st.success("✅ Model trained and saved to session! You can now use the Transaction Scorer.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: BATCH LEARNING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔄 Batch Learning":
    st.markdown('<div class="phase-badge">PHASE 5 — CONTINUOUS LEARNING</div>', unsafe_allow_html=True)
    st.title("Incremental / Batch Learning")

    if "xgb_model" not in st.session_state:
        st.warning("⬅️  Please train the XGBoost model first (Phase 4 page).")
        st.stop()

    import xgboost as xgb
    from sklearn.metrics import precision_score, recall_score, f1_score

    st.markdown("""
    <p style='color:#94a3b8;'>
    This phase simulates a <strong>new month of transactions</strong> arriving. The existing XGBoost model
    is updated via <strong>warm-start</strong> — adding new trees on top of the current ensemble
    without touching historical data.
    </p>
    """, unsafe_allow_html=True)

    inc_rounds = st.slider("Incremental Boosting Rounds", 10, 100, 50, 10)

    if st.button("🔄 Run Batch Update"):
        df = st.session_state["df_engineered"]
        model = st.session_state["xgb_model"]
        scaler_train = st.session_state["scaler_train"]
        pca_model    = st.session_state["pca_model"]
        kmeans_model = st.session_state["kmeans_model"]
        scaler_full  = st.session_state["scaler_full"]
        threshold    = st.session_state.get("threshold", 0.5)

        with st.spinner("Preparing new batch (last 10% by step)..."):
            new_df = df[df["step"] >= df["step"].quantile(0.90)].copy()
            noise = np.random.normal(0, 0.02 * new_df["amount"].std(), size=len(new_df))
            new_df["amount"] = (new_df["amount"] + noise).clip(lower=0)
            new_df["orig_balance_diff"] = new_df["oldbalanceOrg"] - new_df["newbalanceOrig"]
            new_df["dest_balance_diff"] = new_df["newbalanceDest"] - new_df["oldbalanceDest"]
            new_df["amount_to_orig_balance_ratio"] = np.where(
                new_df["oldbalanceOrg"] > 0, new_df["amount"] / new_df["oldbalanceOrg"], 0)
            new_df["orig_balance_zero_after"]  = (new_df["newbalanceOrig"] == 0).astype(int)
            new_df["dest_balance_zero_before"] = (new_df["oldbalanceDest"] == 0).astype(int)

            new_cont = scaler_full.transform(new_df[CONTINUOUS_FEATURES])
            new_pca  = pca_model.transform(np.hstack([new_cont, new_df[BINARY_FEATURES].values]))
            new_df["behavioral_segment"] = kmeans_model.predict(new_pca)

            new_cont_s = scaler_train.transform(new_df[CONTINUOUS_FEATURES])
            X_new = np.hstack([new_cont_s, new_df[BINARY_FEATURES + ["behavioral_segment"]].values])
            y_new = new_df[TARGET].values
            dnew  = xgb.DMatrix(X_new, label=y_new)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">NEW BATCH SIZE</div>
                <div class="metric-value">{len(new_df):,}</div>
                <div class="metric-sub">transactions</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">FRAUD IN BATCH</div>
                <div class="metric-value" style="color:{RED};">{new_df['isFraud'].sum():,}</div>
                <div class="metric-sub">{new_df['isFraud'].mean()*100:.3f}% rate</div>
            </div>
            """, unsafe_allow_html=True)

        # Evaluate BEFORE
        y_prob_before = model.predict(dnew)
        y_pred_before = (y_prob_before >= threshold).astype(int)
        p_b = precision_score(y_new, y_pred_before, zero_division=0)
        r_b = recall_score(y_new, y_pred_before, zero_division=0)
        f_b = f1_score(y_new, y_pred_before, zero_division=0)

        # Update model
        params = {
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "aucpr"],
            "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8,
            "colsample_bytree": 0.8, "min_child_weight": 5,
            "scale_pos_weight": (y_new == 0).sum() / max(1, (y_new == 1).sum()),
            "seed": SEED, "tree_method": "hist",
        }
        with st.spinner(f"Adding {inc_rounds} new trees..."):
            updated_model = xgb.train(
                params, dnew,
                num_boost_round=inc_rounds,
                xgb_model=model,
                evals=[(dnew, "new_batch")],
                verbose_eval=False,
            )

        # Evaluate AFTER
        y_prob_after = updated_model.predict(dnew)
        y_pred_after = (y_prob_after >= threshold).astype(int)
        p_a = precision_score(y_new, y_pred_after, zero_division=0)
        r_a = recall_score(y_new, y_pred_after, zero_division=0)
        f_a = f1_score(y_new, y_pred_after, zero_division=0)

        st.markdown("### 📊 Before vs After Batch Update")
        metrics   = ["Precision", "Recall", "F1-Score"]
        before_v  = [p_b, r_b, f_b]
        after_v   = [p_a, r_a, f_a]

        col1, col2, col3 = st.columns(3)
        for col, metric, bv, av in zip([col1, col2, col3], metrics, before_v, after_v):
            delta = av - bv
            color = GREEN if delta >= 0 else RED
            arrow = "▲" if delta >= 0 else "▼"
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{metric}</div>
                    <div class="metric-value">{av:.4f}</div>
                    <div class="metric-sub">Before: {bv:.4f}
                        <span style="color:{color}; margin-left:6px;">{arrow} {abs(delta):.4f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(metrics))
        w = 0.35
        bars1 = ax.bar(x - w/2, before_v, w, label="Before Update", color=ACCENT, edgecolor="#0a0e1a")
        bars2 = ax.bar(x + w/2, after_v,  w, label="After Update",  color=GREEN,  edgecolor="#0a0e1a")
        ax.set_ylim(0, 1.1)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylabel("Score")
        ax.set_title("Continuous Learning — Performance Comparison")
        ax.legend()
        for bar in list(bars1) + list(bars2):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
        fig_to_st(fig)

        st.session_state["xgb_model"] = updated_model
        st.success(f"✅ Model updated with {inc_rounds} additional trees. Total rounds ≈ 300 + {inc_rounds}.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: TRANSACTION SCORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧪 Transaction Scorer":
    st.markdown('<div class="phase-badge">LIVE INFERENCE</div>', unsafe_allow_html=True)
    st.title("Transaction Risk Scorer")

    st.markdown("""
    <p style='color:#94a3b8;'>
    Enter transaction details below to get an instant AML risk score.
    If the XGBoost model is trained (Phase 4), it will be used; otherwise a rule-based heuristic runs.
    </p>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        tx_type   = st.selectbox("Transaction Type", ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"])
        amount    = st.number_input("Amount ($)", min_value=0.0, value=50000.0, step=1000.0)
        step      = st.number_input("Step (hour of simulation)", min_value=1, max_value=744, value=100)
    with col2:
        old_orig  = st.number_input("Sender Old Balance ($)", min_value=0.0, value=60000.0, step=1000.0)
        new_orig  = st.number_input("Sender New Balance ($)", min_value=0.0, value=10000.0, step=1000.0)
    with col3:
        old_dest  = st.number_input("Receiver Old Balance ($)", min_value=0.0, value=0.0, step=1000.0)
        new_dest  = st.number_input("Receiver New Balance ($)", min_value=0.0, value=50000.0, step=1000.0)

    if st.button("🔍 Analyze Transaction"):
        # Feature engineering
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
        type_enc = int(le.transform([tx_type])[0])

        orig_diff  = old_orig - new_orig
        dest_diff  = new_dest - old_dest
        ratio      = amount / old_orig if old_orig > 0 else 0.0
        zero_after = int(new_orig == 0)
        zero_before= int(old_dest == 0)

        feat_vec = np.array([[step, amount, old_orig, new_orig,
                               old_dest, new_dest, orig_diff, dest_diff,
                               ratio, type_enc, zero_after, zero_before]])

        model_used = False
        if "xgb_model" in st.session_state:
            import xgboost as xgb
            scaler_train = st.session_state.get("scaler_train")
            pca_model    = st.session_state.get("pca_model")
            kmeans_model = st.session_state.get("kmeans_model")
            scaler_full  = st.session_state.get("scaler_full")
            xgb_model    = st.session_state["xgb_model"]
            threshold    = st.session_state.get("threshold", 0.5)

            if all([scaler_train, pca_model, kmeans_model, scaler_full]):
                cont_s = scaler_full.transform(feat_vec[:, :9])
                pca_in = np.hstack([cont_s, feat_vec[:, 9:12]])
                pca_out = pca_model.transform(pca_in)
                segment = kmeans_model.predict(pca_out)
                cont_ts = scaler_train.transform(feat_vec[:, :9])
                final_feat = np.hstack([cont_ts, feat_vec[:, 9:12], segment.reshape(-1, 1)])
                dm = xgb.DMatrix(final_feat)
                risk_score = float(xgb_model.predict(dm)[0])
                model_used = True

        if not model_used:
            # Rule-based heuristic
            risk_score = 0.0
            if tx_type in ["CASH_OUT", "TRANSFER"]:
                risk_score += 0.30
            if ratio > 0.9:
                risk_score += 0.25
            if zero_after:
                risk_score += 0.20
            if zero_before:
                risk_score += 0.10
            if amount > 200000:
                risk_score += 0.15
            risk_score = min(risk_score, 0.99)

        # Display result
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        col_score, col_detail = st.columns([1, 2])

        with col_score:
            pct = int(risk_score * 100)
            color = RED if risk_score >= 0.5 else (ORANGE if risk_score >= 0.25 else GREEN)
            label = "🚨 HIGH RISK" if risk_score >= 0.5 else ("⚠️ MEDIUM RISK" if risk_score >= 0.25 else "✅ LOW RISK")
            st.markdown(f"""
            <div class="metric-card" style="border-color:{color};">
                <div class="metric-label">FRAUD PROBABILITY</div>
                <div class="metric-value" style="color:{color}; font-size:3rem;">{pct}%</div>
                <div style="font-size:1rem; color:{color}; font-weight:700; margin-top:8px;">{label}</div>
                <div class="metric-sub">{"XGBoost model" if model_used else "Rule-based heuristic"}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_detail:
            alert_class = "alert-fraud" if risk_score >= 0.5 else "alert-safe"
            flags = []
            if tx_type in ["CASH_OUT", "TRANSFER"]: flags.append("Suspicious transaction type")
            if ratio > 0.9:   flags.append(f"Account drain ratio: {ratio:.1%}")
            if zero_after:    flags.append("Sender balance zeroed after transaction")
            if zero_before:   flags.append("Receiver had zero balance before transaction")
            if amount > 200000: flags.append(f"Large amount: ${amount:,.0f}")
            if not flags:     flags.append("No significant risk flags detected")

            flags_html = "".join([f"<div>• {f}</div>" for f in flags])
            st.markdown(f"""
            <div class="{alert_class}">
                <div style="font-weight:700; margin-bottom:8px;">Risk Factors Detected:</div>
                {flags_html}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Feature Summary")
            summary = pd.DataFrame({
                "Feature": ["Type", "Amount", "Drain Ratio", "Orig Δ Balance",
                             "Dest Δ Balance", "Zero After", "Zero Before"],
                "Value": [tx_type, f"${amount:,.0f}", f"{ratio:.3f}", f"${orig_diff:,.0f}",
                           f"${dest_diff:,.0f}", "Yes" if zero_after else "No",
                           "Yes" if zero_before else "No"],
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)

        # Gauge chart
        fig, ax = plt.subplots(figsize=(5, 3), subplot_kw=dict(projection="polar"))
        theta = np.linspace(0, np.pi, 300)
        ax.plot(theta, [1]*300, color="#1e293b", linewidth=15, solid_capstyle="round")
        filled = int(risk_score * 300)
        fill_color = RED if risk_score >= 0.5 else (ORANGE if risk_score >= 0.25 else GREEN)
        ax.plot(theta[:filled], [1]*filled, color=fill_color, linewidth=15, solid_capstyle="round")
        ax.set_ylim(0, 1.5)
        ax.set_theta_zero_location("W")
        ax.set_theta_direction(-1)
        ax.set_axis_off()
        ax.text(np.pi/2, 0.3, f"{pct}%", ha="center", va="center",
                fontsize=22, fontweight="bold", color=fill_color, transform=ax.transData)
        ax.text(np.pi/2, 0.05, "RISK SCORE", ha="center", va="center",
                fontsize=8, color="#64748b", transform=ax.transData)
        fig_to_st(fig)
