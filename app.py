"""
app.py — Idealista Investment Opportunity Detector

Run from the project root:

    python -m streamlit run app.py

This version is designed to be visually clean, stable, and functional:
- light dashboard theme
- custom sidebar
- white chart backgrounds
- robust data loading
- graceful fallback if some files are missing
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =============================================================================
# Page config
# =============================================================================

st.set_page_config(
    page_title="Idealista Investment Detector",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

# =============================================================================
# CSS — clean SaaS / Figma-inspired style
# =============================================================================

st.markdown(
    """
    <style>
    /* Global background */
    .stApp {
        background: #f8fafc;
        color: #0f172a;
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    /* Hide default Streamlit top decoration */
    [data-testid="stDecoration"] {
        display: none;
    }

    /* Main content spacing */
    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.4rem;
    }

    [data-testid="stSidebar"] * {
        color: #334155;
    }

    /* Radio navigation cards */
    [data-testid="stSidebar"] [role="radiogroup"] label {
        border-radius: 12px;
        padding: 0.65rem 0.75rem;
        margin-bottom: 0.25rem;
        background: transparent;
        transition: all 0.15s ease;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: #f1f5f9;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label[data-baseweb="radio"] {
        font-weight: 600;
    }

    /* Typography */
    h1, h2, h3 {
        color: #0f172a !important;
        letter-spacing: -0.025em;
    }

    p, li, span, div {
        color: inherit;
    }

    /* Custom components */
    .app-logo {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.25rem 0.2rem 1.1rem 0.2rem;
        margin-bottom: 0.8rem;
        border-bottom: 1px solid #e2e8f0;
    }

    .logo-icon {
        width: 42px;
        height: 42px;
        border-radius: 14px;
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 22px;
        box-shadow: 0 10px 24px rgba(37, 99, 235, 0.22);
    }

    .logo-title {
        font-weight: 800;
        font-size: 1.05rem;
        color: #0f172a;
        line-height: 1.1;
    }

    .logo-subtitle {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 0.15rem;
    }

    .side-note {
        background: linear-gradient(135deg, #eff6ff 0%, #eef2ff 100%);
        border: 1px solid #dbeafe;
        border-radius: 16px;
        padding: 1rem;
        margin-top: 1.4rem;
        color: #334155;
    }

    .side-note strong {
        color: #1e3a8a;
    }

    .hero {
        background: linear-gradient(135deg, #ffffff 0%, #eff6ff 100%);
        border: 1px solid #dbeafe;
        border-radius: 24px;
        padding: 2rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 20px 50px rgba(15, 23, 42, 0.06);
    }

    .hero-small {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #2563eb;
        font-weight: 800;
        margin-bottom: 0.55rem;
    }

    .hero h1 {
        font-size: 2.1rem;
        margin-bottom: 0.45rem;
        line-height: 1.15;
    }

    .hero p {
        font-size: 1rem;
        color: #475569;
        margin-bottom: 0;
        max-width: 850px;
    }

    .card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 1.25rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.045);
        margin-bottom: 1rem;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 1.15rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.045);
        min-height: 122px;
    }

    .metric-label {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.55rem;
    }

    .metric-value {
        color: #0f172a;
        font-size: 1.85rem;
        font-weight: 850;
        line-height: 1.05;
    }

    .metric-help {
        color: #64748b;
        font-size: 0.82rem;
        margin-top: 0.45rem;
    }

    .blue-card {
        background: linear-gradient(135deg, #eff6ff 0%, #eef2ff 100%);
        border: 1px solid #bfdbfe;
    }

    .green-card {
        background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
        border: 1px solid #bbf7d0;
    }

    .amber-card {
        background: linear-gradient(135deg, #fffbeb 0%, #fefce8 100%);
        border: 1px solid #fde68a;
    }

    .section-title {
        color: #0f172a;
        font-size: 1.15rem;
        font-weight: 800;
        margin: 0 0 0.9rem 0;
    }

    .muted {
        color: #64748b;
    }

    .pipeline {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
        align-items: center;
        margin-top: 0.8rem;
    }

    .pipeline-step {
        background: #ffffff;
        border: 1px solid #dbeafe;
        color: #1e3a8a;
        border-radius: 999px;
        padding: 0.55rem 0.85rem;
        font-size: 0.85rem;
        font-weight: 750;
        box-shadow: 0 8px 18px rgba(37, 99, 235, 0.06);
    }

    .status-badge {
        display: inline-block;
        border-radius: 999px;
        padding: 0.35rem 0.7rem;
        font-size: 0.78rem;
        font-weight: 800;
        background: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
    }

    .warning-badge {
        display: inline-block;
        border-radius: 999px;
        padding: 0.35rem 0.7rem;
        font-size: 0.78rem;
        font-weight: 800;
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #fde68a;
    }

    .danger-badge {
        display: inline-block;
        border-radius: 999px;
        padding: 0.35rem 0.7rem;
        font-size: 0.78rem;
        font-weight: 800;
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }

    /* Widgets */
    div[data-testid="stSelectbox"] > div,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
        border-radius: 12px !important;
    }

    .stButton > button {
        border-radius: 14px;
        border: 0;
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
        color: white;
        font-weight: 800;
        padding: 0.7rem 1.15rem;
        box-shadow: 0 12px 24px rgba(37, 99, 235, 0.22);
    }

    .stButton > button:hover {
        border: 0;
        color: white;
        filter: brightness(0.98);
        transform: translateY(-1px);
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
        background: white;
    }

    /* Matplotlib figure container should be white */
    .stPlotlyChart, [data-testid="stImage"], [data-testid="stPyplot"] {
        background: #ffffff !important;
        border-radius: 18px;
    }

    hr {
        border-color: #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Helpers
# =============================================================================

def find_latest_file(directory: Path, pattern: str) -> Path | None:
    if not directory.exists():
        return None
    files = sorted(directory.glob(pattern))
    return files[-1] if files else None


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    import joblib
    return joblib.load(path)


def read_available_data() -> tuple[dict[str, Path | None], dict[str, pd.DataFrame]]:
    paths = {
        "raw": find_latest_file(RAW_DATA_DIR, "idealista_raw_listings_*.csv"),
        "clean": find_latest_file(PROCESSED_DATA_DIR, "idealista_clean_full_*.csv"),
        "model_ready": PROCESSED_DATA_DIR / "idealista_model_ready_latest.csv",
        "predictions": PROCESSED_DATA_DIR / "model_predictions_latest.csv",
        "metrics": REPORTS_DIR / "model_metrics_latest.csv",
        "model": MODELS_DIR / "best_model_latest.pkl",
    }

    data: dict[str, pd.DataFrame] = {}
    for key, path in paths.items():
        if key == "model":
            continue
        if path is not None and path.exists():
            try:
                data[key] = load_csv(str(path))
            except Exception as exc:
                st.warning(f"Could not load {path}: {exc}")

    return paths, data


def fmt_eur(value, decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"€{value:,.{decimals}f}"


def fmt_num(value, suffix: str = "", decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}{suffix}"


def metric_card(label: str, value: str, help_text: str = "", style: str = ""):
    st.markdown(
        f"""
        <div class="metric-card {style}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, eyebrow: str = "Machine Learning Real Estate Project"):
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-small">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card_start(title: str | None = None, subtitle: str | None = None):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if title:
        st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="muted" style="margin-top:-0.55rem; margin-bottom:1rem;">{subtitle}</div>', unsafe_allow_html=True)


def card_end():
    st.markdown("</div>", unsafe_allow_html=True)


def make_white_fig(figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.tick_params(colors="#475569")
    ax.xaxis.label.set_color("#334155")
    ax.yaxis.label.set_color("#334155")
    ax.title.set_color("#0f172a")
    for spine in ax.spines.values():
        spine.set_color("#e2e8f0")
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def show_histogram(series: pd.Series, title: str, xlabel: str, bins: int = 35):
    values = pd.to_numeric(series, errors="coerce").dropna()
    fig, ax = make_white_fig((10, 5))
    ax.hist(values, bins=bins, color="#2563eb", edgecolor="white", linewidth=0.5)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of listings")
    ax.ticklabel_format(style="plain", axis="x")
    st.pyplot(fig, clear_figure=True)


def show_bar(series: pd.Series, title: str, ylabel: str, rotation: int = 30):
    fig, ax = make_white_fig((10, 5.2))
    ax.bar(series.index.astype(str), series.values, color="#2563eb")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotation)
    ax.ticklabel_format(style="plain", axis="y")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def show_scatter(df: pd.DataFrame, x: str, y: str, title: str):
    plot_df = df[[x, y]].dropna().copy()
    fig, ax = make_white_fig((7.5, 7.5))
    ax.scatter(plot_df[x], plot_df[y], alpha=0.65, color="#2563eb", edgecolors="white", linewidths=0.35)
    min_v = min(plot_df[x].min(), plot_df[y].min())
    max_v = max(plot_df[x].max(), plot_df[y].max())
    ax.plot([min_v, max_v], [min_v, max_v], color="#ef4444", linestyle="--", linewidth=1.5)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Actual price (€)")
    ax.set_ylabel("Predicted price (€)")
    ax.ticklabel_format(style="plain", axis="both")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def opportunity_score(row: pd.Series) -> float:
    if "predicted_price" not in row or "actual_price" not in row:
        return np.nan
    actual = row["actual_price"]
    predicted = row["predicted_price"]
    if pd.isna(actual) or actual <= 0 or pd.isna(predicted):
        return np.nan
    gap_pct = (predicted - actual) / actual * 100
    return float(np.clip(50 + gap_pct * 2.5, 0, 100))


def add_opportunity_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"actual_price", "predicted_price"}.issubset(out.columns):
        out["price_gap"] = out["predicted_price"] - out["actual_price"]
        out["percentage_error"] = out["price_gap"] / out["actual_price"] * 100
        out["opportunity_score"] = out.apply(opportunity_score, axis=1)
        out["valuation_status"] = np.select(
            [out["percentage_error"] >= 10, out["percentage_error"] <= -10],
            ["Potentially undervalued", "Potentially overvalued"],
            default="Fairly priced",
        )
    return out


def compatible_width():
    # Streamlit new versions prefer width="stretch"; this helper keeps code readable.
    return {"width": "stretch"}


# =============================================================================
# Load data
# =============================================================================

paths, data = read_available_data()

raw_df = data.get("raw")
clean_df = data.get("clean")
model_df = data.get("model_ready")
predictions_df = data.get("predictions")
metrics_df = data.get("metrics")

if predictions_df is not None:
    predictions_df = add_opportunity_columns(predictions_df)

# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown(
        """
        <div class="app-logo">
            <div class="logo-icon">🏠</div>
            <div>
                <div class="logo-title">Idealista</div>
                <div class="logo-subtitle">Investment Detector</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Project Overview",
            "Raw Data",
            "Model Performance",
            "Opportunity Explorer",
            "Manual Estimator",
            "Data Health",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        """
        <div class="side-note">
            <strong>ML Real Estate Project</strong><br/>
            <span style="font-size:0.82rem;color:#64748b;">
            Madrid premium residential properties.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# Page: Project Overview
# =============================================================================

if page == "Project Overview":
    hero(
        "Idealista Investment Opportunity Detector",
        "AI-powered real estate price estimation and investment opportunity scoring for Madrid premium residential properties.",
    )

    c1, c2, c3, c4 = st.columns(4)

    rows = len(model_df) if model_df is not None else 0
    features = max(model_df.shape[1] - 1, 0) if model_df is not None else 0
    median_price = model_df["price"].median() if model_df is not None and "price" in model_df.columns else np.nan

    best_r2 = np.nan
    if metrics_df is not None and "test_r2" in metrics_df.columns:
        best_r2 = metrics_df["test_r2"].max()

    with c1:
        metric_card("Model-ready listings", fmt_num(rows), "Clean properties")
    with c2:
        metric_card("Features engineered", fmt_num(features), "Predictive variables")
    with c3:
        metric_card("Median property price", fmt_eur(median_price), "Premium segment")
    with c4:
        metric_card("Best model R²", fmt_num(best_r2, decimals=2), "Gradient Boosting", "blue-card")

    card_start("Pipeline Overview")
    st.markdown(
        """
        <div class="pipeline">
            <div class="pipeline-step">1. Idealista API</div>
            <div class="pipeline-step">2. Raw Dataset</div>
            <div class="pipeline-step">3. Cleaning</div>
            <div class="pipeline-step">4. Feature Engineering</div>
            <div class="pipeline-step">5. ML Model</div>
            <div class="pipeline-step">6. Investment Score</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    card_end()

    col_a, col_b = st.columns(2)

    with col_a:
        card_start("Project Description")
        st.markdown(
            """
            This machine learning project analyzes Madrid premium residential real estate listings
            to estimate fair market values and identify potentially undervalued investment opportunities.

            The model compares predicted fair prices against actual listing prices to generate an
            investment opportunity signal.
            """
        )
        card_end()

    with col_b:
        card_start("Key Insights")
        st.markdown(
            """
            - **Gradient Boosting** is the best-performing model.
            - **Size and location** are the strongest price drivers.
            - The app allows dynamic filtering and manual estimation.
            - The project is connected to real model outputs, not only mock data.
            """
        )
        card_end()

    card_start("Repository File Status")
    status_rows = []
    labels = {
        "raw": "Raw API data",
        "clean": "Clean full dataset",
        "model_ready": "Model-ready dataset",
        "predictions": "Model predictions",
        "metrics": "Model metrics",
        "model": "Best model pickle",
    }
    for key, label in labels.items():
        path = paths.get(key)
        status_rows.append(
            {
                "File": label,
                "Available": bool(path is not None and Path(path).exists()),
                "Path": str(path) if path is not None else "Not found",
            }
        )
    st.dataframe(pd.DataFrame(status_rows), hide_index=True, **compatible_width())
    card_end()

# =============================================================================
# Page: Raw Data
# =============================================================================

elif page == "Raw Data":
    hero(
        "Raw Data Exploration",
        "Explore the Idealista listings collected from the API before and after the first processing steps.",
        "Dataset Exploration",
    )

    df_display = raw_df if raw_df is not None else model_df

    if df_display is None:
        st.error("No dataset found. Run the data collection and feature engineering notebooks first.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Rows", fmt_num(len(df_display)), "Loaded observations")
        with c2:
            metric_card("Columns", fmt_num(df_display.shape[1]), "Available variables")
        with c3:
            price_val = pd.to_numeric(df_display.get("price"), errors="coerce").median() if "price" in df_display.columns else np.nan
            metric_card("Median price", fmt_eur(price_val), "Raw listing prices")
        with c4:
            dist_count = df_display["district"].nunique() if "district" in df_display.columns else np.nan
            metric_card("Districts", fmt_num(dist_count), "Market coverage", "blue-card")

        card_start("Filters")
        f1, f2, f3 = st.columns(3)
        filtered = df_display.copy()

        if "district" in filtered.columns:
            districts = ["All"] + sorted(filtered["district"].dropna().astype(str).unique().tolist())
            selected_district = f1.selectbox("District", districts)
            if selected_district != "All":
                filtered = filtered[filtered["district"].astype(str) == selected_district]

        if "propertyType" in filtered.columns:
            property_types = ["All"] + sorted(filtered["propertyType"].dropna().astype(str).unique().tolist())
            selected_type = f2.selectbox("Property type", property_types)
            if selected_type != "All":
                filtered = filtered[filtered["propertyType"].astype(str) == selected_type]

        if "price" in filtered.columns:
            numeric_price = pd.to_numeric(filtered["price"], errors="coerce")
            max_price_default = int(numeric_price.quantile(0.95)) if numeric_price.notna().any() else 10_000_000
            max_price = f3.number_input("Max price (€)", min_value=0, value=max_price_default, step=100_000)
            filtered = filtered[pd.to_numeric(filtered["price"], errors="coerce") <= max_price]
        card_end()

        col1, col2 = st.columns(2)
        with col1:
            card_start("Raw Price Distribution")
            if "price" in filtered.columns:
                show_histogram(filtered["price"], "Distribution of Listing Prices", "Price (€)")
            else:
                st.info("Price column not found.")
            card_end()

        with col2:
            card_start("Median Price by District")
            if {"district", "price"}.issubset(filtered.columns):
                district_price = (
                    filtered.assign(price=pd.to_numeric(filtered["price"], errors="coerce"))
                    .dropna(subset=["price", "district"])
                    .groupby("district")["price"]
                    .median()
                    .sort_values(ascending=False)
                    .head(12)
                )
                show_bar(district_price, "Median Price by District", "Median price (€)", rotation=45)
            else:
                st.info("District or price column not found.")
            card_end()

        card_start("Dataset Preview")
        st.dataframe(filtered.head(80), hide_index=True, **compatible_width())
        card_end()

# =============================================================================
# Page: Model Performance
# =============================================================================

elif page == "Model Performance":
    hero(
        "Model Performance",
        "Compare the three trained models and inspect prediction quality.",
        "Model Evaluation",
    )

    if metrics_df is None:
        st.error("No model metrics file found. Run notebooks/04_model_training.ipynb first.")
    else:
        metrics = metrics_df.copy()

        if "test_mae" in metrics.columns:
            best_row = metrics.sort_values("test_mae").iloc[0]
        else:
            best_row = metrics.iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Best model", str(best_row.get("model", "N/A")), "Selected by lowest MAE", "blue-card")
        with c2:
            metric_card("Test MAE", fmt_eur(best_row.get("test_mae")), "Average absolute error")
        with c3:
            metric_card("Test RMSE", fmt_eur(best_row.get("test_rmse")), "Penalizes large errors")
        with c4:
            metric_card("Test R²", fmt_num(best_row.get("test_r2"), decimals=2), "Explained variance", "green-card")

        card_start("Metrics Table")
        st.dataframe(metrics, hide_index=True, **compatible_width())
        card_end()

        col1, col2 = st.columns(2)

        with col1:
            card_start("Model Comparison — Test MAE")
            if {"model", "test_mae"}.issubset(metrics.columns):
                mae_series = metrics.sort_values("test_mae").set_index("model")["test_mae"]
                show_bar(mae_series, "Test MAE by Model", "MAE (€)", rotation=30)
            else:
                st.info("Required metric columns not found.")
            card_end()

        with col2:
            card_start("Model Comparison — Test R²")
            if {"model", "test_r2"}.issubset(metrics.columns):
                r2_series = metrics.sort_values("test_r2", ascending=False).set_index("model")["test_r2"]
                show_bar(r2_series, "Test R² by Model", "R²", rotation=30)
            else:
                st.info("Required metric columns not found.")
            card_end()

        card_start("Actual vs Predicted Prices")
        if predictions_df is not None and {"actual_price", "predicted_price"}.issubset(predictions_df.columns):
            show_scatter(predictions_df, "actual_price", "predicted_price", "Actual vs Predicted Property Prices")
        else:
            st.info("Predictions file not found or missing actual_price/predicted_price columns.")
        card_end()

# =============================================================================
# Page: Opportunity Explorer
# =============================================================================

elif page == "Opportunity Explorer":
    hero(
        "Opportunity Explorer",
        "Rank properties by model-estimated upside and identify potential investment opportunities.",
        "Investment Scoring",
    )

    if predictions_df is None:
        st.error("No predictions file found. Run notebooks/04_model_training.ipynb first.")
    else:
        opp = predictions_df.copy()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Predictions", fmt_num(len(opp)), "Test-set properties")
        with c2:
            metric_card("Top score", fmt_num(opp["opportunity_score"].max(), decimals=0), "Best detected opportunity", "blue-card")
        with c3:
            avg_upside = opp["percentage_error"].mean() if "percentage_error" in opp.columns else np.nan
            metric_card("Avg. predicted gap", fmt_num(avg_upside, "%", 1), "Predicted vs actual")
        with c4:
            undervalued_count = (opp["valuation_status"] == "Potentially undervalued").sum() if "valuation_status" in opp.columns else 0
            metric_card("Potential opportunities", fmt_num(undervalued_count), "Predicted upside > 10%", "green-card")

        card_start("Filters")
        f1, f2, f3, f4 = st.columns(4)

        min_score = f1.slider("Minimum score", 0, 100, 55)
        max_price = f2.number_input(
            "Maximum actual price (€)",
            min_value=0,
            value=int(opp["actual_price"].max()) if "actual_price" in opp.columns else 10_000_000,
            step=100_000,
        )

        if "district" in opp.columns:
            districts = ["All"] + sorted(opp["district"].dropna().astype(str).unique().tolist())
            selected_district = f3.selectbox("District", districts)
        else:
            selected_district = "All"

        if "propertyType" in opp.columns:
            types = ["All"] + sorted(opp["propertyType"].dropna().astype(str).unique().tolist())
            selected_type = f4.selectbox("Property type", types)
        else:
            selected_type = "All"
        card_end()

        filtered = opp.copy()
        if "opportunity_score" in filtered.columns:
            filtered = filtered[filtered["opportunity_score"] >= min_score]
        if "actual_price" in filtered.columns:
            filtered = filtered[filtered["actual_price"] <= max_price]
        if selected_district != "All" and "district" in filtered.columns:
            filtered = filtered[filtered["district"].astype(str) == selected_district]
        if selected_type != "All" and "propertyType" in filtered.columns:
            filtered = filtered[filtered["propertyType"].astype(str) == selected_type]

        filtered = filtered.sort_values("opportunity_score", ascending=False)

        card_start("Top Potential Opportunities")
        display_cols = [
            "opportunity_score",
            "valuation_status",
            "actual_price",
            "predicted_price",
            "percentage_error",
            "district",
            "neighborhood",
            "propertyType",
            "size",
            "rooms",
            "bathrooms",
        ]
        display_cols = [col for col in display_cols if col in filtered.columns]
        st.dataframe(filtered[display_cols].head(30), hide_index=True, **compatible_width())
        card_end()

        if not filtered.empty:
            card_start("Selected Property Detail")
            selected_rank = st.selectbox("Choose a property rank", list(range(1, min(30, len(filtered)) + 1)))
            row = filtered.iloc[selected_rank - 1]

            d1, d2, d3, d4 = st.columns(4)
            with d1:
                metric_card("Actual price", fmt_eur(row.get("actual_price")), "Current listing")
            with d2:
                metric_card("Predicted fair price", fmt_eur(row.get("predicted_price")), "Model estimate")
            with d3:
                metric_card("Predicted gap", fmt_num(row.get("percentage_error"), "%", 1), "Upside/downside")
            with d4:
                metric_card("Opportunity score", fmt_num(row.get("opportunity_score"), "/100", 0), "Business score", "blue-card")

            status = row.get("valuation_status", "Unknown")
            if status == "Potentially undervalued":
                badge = "status-badge"
            elif status == "Potentially overvalued":
                badge = "danger-badge"
            else:
                badge = "warning-badge"

            st.markdown(f'<span class="{badge}">{status}</span>', unsafe_allow_html=True)
            st.markdown("")
            st.markdown(
                """
                This score is based on the difference between model-predicted fair price and actual listing price.
                It is a first analytical filter, not a final investment recommendation.
                """
            )

            details = row.to_frame("value")
            st.dataframe(details, **compatible_width())
            card_end()

# =============================================================================
# Page: Manual Estimator
# =============================================================================

elif page == "Manual Estimator":
    hero(
        "Manual Fair Price Estimator",
        "Enter property characteristics and compare the estimated fair market price with a listing price.",
        "Interactive Prediction",
    )

    if model_df is None:
        st.error("No model-ready dataset found. Run notebooks/03_feature_engineering.ipynb first.")
    else:
        model = None
        model_path = paths.get("model")
        if model_path is not None and model_path.exists():
            try:
                model = load_model(str(model_path))
            except Exception as exc:
                st.warning(f"Could not load the trained model. Fallback estimator will be used. Error: {exc}")
        else:
            st.warning("No trained model found. Fallback estimator will be used.")

        card_start("Property Inputs")
        c1, c2, c3 = st.columns(3)
        size = c1.number_input("Size (m²)", min_value=20, max_value=1000, value=120, step=5)
        rooms = c2.number_input("Rooms", min_value=0, max_value=10, value=3, step=1)
        bathrooms = c3.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)

        c4, c5, c6 = st.columns(3)

        districts = sorted(model_df["district"].dropna().astype(str).unique()) if "district" in model_df.columns else ["Barrio de Salamanca"]
        property_types = sorted(model_df["propertyType"].dropna().astype(str).unique()) if "propertyType" in model_df.columns else ["flat"]

        district = c4.selectbox("District", districts)
        property_type = c5.selectbox("Property type", property_types)
        asking_price = c6.number_input("Actual listing price (€)", min_value=0, value=1_000_000, step=50_000)

        c7, c8, c9 = st.columns(3)
        has_lift = c7.checkbox("Has lift", value=True)
        has_parking = c8.checkbox("Has parking", value=False)
        is_exterior = c9.checkbox("Exterior", value=True)

        estimate_button = st.button("Estimate fair price")
        card_end()

        if estimate_button:
            feature_df = model_df.drop(columns=["price"], errors="ignore")
            defaults = {}
            for col in feature_df.columns:
                if pd.api.types.is_numeric_dtype(feature_df[col]):
                    defaults[col] = feature_df[col].median()
                else:
                    mode = feature_df[col].mode()
                    defaults[col] = mode.iloc[0] if len(mode) else "Unknown"

            input_row = pd.DataFrame([defaults])

            updates = {
                "size": size,
                "rooms": rooms,
                "bathrooms": bathrooms,
                "district": district,
                "propertyType": property_type,
                "has_lift": int(has_lift),
                "has_parking": int(has_parking),
                "is_exterior": int(is_exterior),
                "rooms_per_100m2": rooms / size * 100 if size else 0,
                "bathrooms_per_room": bathrooms / rooms if rooms else bathrooms,
                "log_size": np.log1p(size),
            }

            if "numPhotos" in input_row.columns and "log_num_photos" in input_row.columns:
                input_row["log_num_photos"] = np.log1p(input_row["numPhotos"])

            for col, val in updates.items():
                if col in input_row.columns:
                    input_row[col] = val

            if model is not None:
                try:
                    predicted_price = float(model.predict(input_row)[0])
                except Exception as exc:
                    st.error(f"Prediction failed. Fallback estimator used. Error: {exc}")
                    predicted_price = np.nan
            else:
                predicted_price = np.nan

            # Fallback if model unavailable or prediction failed
            if pd.isna(predicted_price):
                district_rows = model_df[model_df["district"].astype(str) == district] if "district" in model_df.columns else model_df
                if len(district_rows) > 0 and {"price", "size"}.issubset(district_rows.columns):
                    median_price_m2 = (district_rows["price"] / district_rows["size"]).median()
                else:
                    median_price_m2 = (model_df["price"] / model_df["size"]).median()
                predicted_price = float(median_price_m2 * size)

            gap = predicted_price - asking_price
            gap_pct = gap / asking_price * 100 if asking_price else np.nan
            score = float(np.clip(50 + gap_pct * 2.5, 0, 100)) if not pd.isna(gap_pct) else np.nan

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                metric_card("Estimated fair price", fmt_eur(predicted_price), "Model or fallback")
            with c2:
                metric_card("Actual listing price", fmt_eur(asking_price), "User input")
            with c3:
                metric_card("Gap", fmt_num(gap_pct, "%", 1), "Predicted vs listed")
            with c4:
                metric_card("Opportunity score", fmt_num(score, "/100", 0), "Investment signal", "blue-card")

            if gap_pct >= 10:
                st.markdown('<span class="status-badge">Potentially undervalued</span>', unsafe_allow_html=True)
            elif gap_pct <= -10:
                st.markdown('<span class="danger-badge">Potentially overvalued</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="warning-badge">Fairly priced</span>', unsafe_allow_html=True)

# =============================================================================
# Page: Data Health
# =============================================================================

elif page == "Data Health":
    hero(
        "Data Health",
        "Inspect missing values, column types and dataset quality.",
        "Technical Quality Report",
    )

    available_names = [name for name in ["raw", "clean", "model_ready", "predictions", "metrics"] if data.get(name) is not None]
    if not available_names:
        st.error("No datasets available.")
    else:
        selected = st.selectbox("Dataset", available_names)
        selected_df = data[selected]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Rows", fmt_num(len(selected_df)), "Dataset size")
        with c2:
            metric_card("Columns", fmt_num(selected_df.shape[1]), "Variables")
        with c3:
            missing_total = int(selected_df.isna().sum().sum())
            metric_card("Missing values", fmt_num(missing_total), "Total missing cells")
        with c4:
            duplicate_total = int(selected_df.duplicated().sum())
            metric_card("Exact duplicates", fmt_num(duplicate_total), "Duplicate rows", "green-card")

        health = pd.DataFrame(
            {
                "column": selected_df.columns,
                "dtype": selected_df.dtypes.astype(str).values,
                "missing_count": selected_df.isna().sum().values,
                "missing_pct": (selected_df.isna().mean().values * 100).round(2),
                "unique_values": selected_df.nunique(dropna=True).values,
            }
        ).sort_values("missing_pct", ascending=False)

        card_start("Column Health")
        st.dataframe(health, hide_index=True, **compatible_width())
        card_end()

        card_start("Dataset Preview")
        st.dataframe(selected_df.head(80), hide_index=True, **compatible_width())
        card_end()
