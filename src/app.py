from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"


def find_latest_file(directory: Path, pattern: str) -> Path | None:
    if not directory.exists():
        return None
    files = sorted(directory.glob(pattern))
    return files[-1] if files else None


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_pickle_model(path: str):
    import joblib
    return joblib.load(path)


def get_first_available(data: dict, keys: list[str]):
    """
    Return the first non-None object from a dictionary.
    This avoids using `or` on pandas DataFrames.
    """
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return None

def load_project_data() -> tuple[dict[str, Path | None], dict[str, pd.DataFrame]]:
    paths = {
        "raw": find_latest_file(RAW_DATA_DIR, "idealista_raw_listings_*.csv"),
        "clean": find_latest_file(PROCESSED_DATA_DIR, "idealista_clean_full_*.csv"),
        "model_ready": PROCESSED_DATA_DIR / "idealista_model_ready_latest.csv",
        "predictions": PROCESSED_DATA_DIR / "model_predictions_latest.csv",
        "metrics": REPORTS_DIR / "model_metrics_latest.csv",
        "main_metrics": PROJECT_ROOT / "results" / "model_metrics.csv",
        "model": MODELS_DIR / "best_model_latest.pkl",
    }

    data: dict[str, pd.DataFrame] = {}
    for key, path in paths.items():
        if key == "model":
            continue
        if path is not None and path.exists():
            try:
                data[key] = load_csv(str(path))
            except Exception:
                pass

    return paths, data


def fmt_eur(value, decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"€{value:,.{decimals}f}"


def fmt_num(value, suffix: str = "", decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}{suffix}"


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f8fafc; color: #0f172a; font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
        [data-testid="stDecoration"] { display: none; }
        .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1380px; }
        [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e2e8f0; }
        [data-testid="stSidebar"] * { color: #334155; }
        h1, h2, h3 { color: #0f172a !important; letter-spacing: -0.025em; }
        .hero { background: linear-gradient(135deg, #ffffff 0%, #eff6ff 100%); border: 1px solid #dbeafe; border-radius: 24px; padding: 2rem; margin-bottom: 1.2rem; box-shadow: 0 20px 50px rgba(15, 23, 42, 0.06); }
        .hero-small { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; color: #2563eb; font-weight: 800; margin-bottom: 0.5rem; }
        .hero h1 { font-size: 2.1rem; margin-bottom: 0.45rem; line-height: 1.15; }
        .hero p { font-size: 1rem; color: #475569; margin-bottom: 0; max-width: 850px; }
        .metric-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 20px; padding: 1.15rem; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.045); min-height: 120px; margin-bottom: 1rem; }
        .metric-label { color: #64748b; font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.55rem; }
        .metric-value { color: #0f172a; font-size: 1.75rem; font-weight: 850; line-height: 1.05; }
        .metric-help { color: #64748b; font-size: 0.82rem; margin-top: 0.45rem; }
        .blue-card { background: linear-gradient(135deg, #eff6ff 0%, #eef2ff 100%); border: 1px solid #bfdbfe; }
        .card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 20px; padding: 1.25rem; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.045); margin-bottom: 1rem; }
        .app-logo { display: flex; align-items: center; gap: 0.75rem; padding-bottom: 1rem; margin-bottom: 1rem; border-bottom: 1px solid #e2e8f0; }
        .logo-icon { width: 42px; height: 42px; border-radius: 14px; background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%); display: flex; align-items: center; justify-content: center; color: white; font-size: 22px; box-shadow: 0 10px 24px rgba(37, 99, 235, 0.22); }
        .logo-title { font-weight: 800; font-size: 1.05rem; color: #0f172a; line-height: 1.1; }
        .logo-subtitle { font-size: 0.75rem; color: #64748b; margin-top: 0.15rem; }
        .stButton > button { border-radius: 14px; border: 0; background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%); color: white; font-weight: 800; padding: 0.7rem 1.15rem; box-shadow: 0 12px 24px rgba(37, 99, 235, 0.22); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, help_text: str = "", style: str = "") -> None:
    st.markdown(f"""<div class="metric-card {style}"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-help">{help_text}</div></div>""", unsafe_allow_html=True)


def hero(title: str, subtitle: str, eyebrow: str = "Machine Learning Real Estate Project") -> None:
    st.markdown(f"""<div class="hero"><div class="hero-small">{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div>""", unsafe_allow_html=True)


def make_white_fig(figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.tick_params(colors="#475569")
    for spine in ax.spines.values():
        spine.set_color("#e2e8f0")
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def show_histogram(series: pd.Series, title: str, xlabel: str) -> None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    fig, ax = make_white_fig((10, 5))
    ax.hist(values, bins=35, color="#2563eb", edgecolor="white", linewidth=0.5)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of listings")
    ax.ticklabel_format(style="plain", axis="x")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def show_bar(series: pd.Series, title: str, ylabel: str) -> None:
    fig, ax = make_white_fig((10, 5.2))
    ax.bar(series.index.astype(str), series.values, color="#2563eb")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    ax.ticklabel_format(style="plain", axis="y")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def show_scatter(df: pd.DataFrame, x: str, y: str, title: str) -> None:
    plot_df = df[[x, y]].dropna()
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


def add_opportunity_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"actual_price", "predicted_price"}.issubset(out.columns):
        out["price_gap"] = out["predicted_price"] - out["actual_price"]
        out["percentage_error"] = out["price_gap"] / out["actual_price"] * 100
        out["opportunity_score"] = np.clip(50 + out["percentage_error"] * 2.5, 0, 100)
        out["valuation_status"] = np.select(
            [out["percentage_error"] >= 10, out["percentage_error"] <= -10],
            ["Potentially undervalued", "Potentially overvalued"],
            default="Fairly priced",
        )
    return out


def build_app() -> None:
    """Callable Streamlit entry point required by the professor's main.py."""
    st.set_page_config(page_title="Idealista Investment Detector", page_icon="🏠", layout="wide", initial_sidebar_state="expanded")
    inject_css()

    paths, data = load_project_data()
    raw_df = data.get("raw")
    model_df = data.get("model_ready")
    predictions_df = data.get("predictions")
    metrics_df = get_first_available(data, ["metrics", "main_metrics"])
    if predictions_df is not None:
        predictions_df = add_opportunity_columns(predictions_df)

    with st.sidebar:
        st.markdown('<div class="app-logo"><div class="logo-icon">🏠</div><div><div class="logo-title">Idealista</div><div class="logo-subtitle">Investment Detector</div></div></div>', unsafe_allow_html=True)
        page = st.radio("Navigation", ["Project Overview", "Raw Data", "Model Performance", "Opportunity Explorer", "Manual Estimator", "Data Health"], label_visibility="collapsed")
        st.caption("Madrid premium residential properties")

    if page == "Project Overview":
        hero("Idealista Investment Opportunity Detector", "AI-powered real estate price estimation and investment opportunity scoring for Madrid premium residential properties.")
        c1, c2, c3, c4 = st.columns(4)
        rows = len(model_df) if model_df is not None else 0
        features = max(model_df.shape[1] - 1, 0) if model_df is not None else 0
        median_price = model_df["price"].median() if model_df is not None and "price" in model_df.columns else np.nan
        best_r2 = metrics_df["test_r2"].max() if metrics_df is not None and "test_r2" in metrics_df.columns else (metrics_df["r2"].max() if metrics_df is not None and "r2" in metrics_df.columns else np.nan)
        with c1: metric_card("Model-ready listings", fmt_num(rows), "Clean properties")
        with c2: metric_card("Features engineered", fmt_num(features), "Predictive variables")
        with c3: metric_card("Median property price", fmt_eur(median_price), "Premium segment")
        with c4: metric_card("Best model R²", fmt_num(best_r2, decimals=2), "Model evaluation", "blue-card")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Pipeline")
        st.markdown("**Idealista API → Raw Dataset → Cleaning → Feature Engineering → ML Model → Investment Score**")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("File Status")
        st.dataframe(pd.DataFrame([{"File": k, "Available": bool(v is not None and Path(v).exists()), "Path": str(v) if v else "Not found"} for k, v in paths.items()]), hide_index=True, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "Raw Data":
        hero("Raw Data Exploration", "Explore the Idealista listings collected from the API.", "Dataset Exploration")
        df_display = raw_df if raw_df is not None else model_df
        if df_display is None:
            st.error("No dataset found.")
            return
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Rows", fmt_num(len(df_display)), "Loaded observations")
        with c2: metric_card("Columns", fmt_num(df_display.shape[1]), "Available variables")
        with c3: metric_card("Median price", fmt_eur(pd.to_numeric(df_display["price"], errors="coerce").median() if "price" in df_display.columns else np.nan), "Raw listing prices")
        with c4: metric_card("Districts", fmt_num(df_display["district"].nunique() if "district" in df_display.columns else np.nan), "Market coverage", "blue-card")
        col1, col2 = st.columns(2)
        with col1:
            if "price" in df_display.columns: show_histogram(df_display["price"], "Distribution of Listing Prices", "Price (€)")
        with col2:
            if {"district", "price"}.issubset(df_display.columns):
                district_price = df_display.assign(price=pd.to_numeric(df_display["price"], errors="coerce")).dropna(subset=["price", "district"]).groupby("district")["price"].median().sort_values(ascending=False).head(12)
                show_bar(district_price, "Median Price by District", "Median price (€)")
        st.subheader("Dataset Preview")
        st.dataframe(df_display.head(80), hide_index=True, width="stretch")

    elif page == "Model Performance":
        hero("Model Performance", "Compare trained models and inspect prediction quality.", "Model Evaluation")
        if metrics_df is None:
            st.error("No model metrics file found.")
            return
        metrics = metrics_df.copy()
        sort_col = "test_mae" if "test_mae" in metrics.columns else ("mae" if "mae" in metrics.columns else None)
        best_row = metrics.sort_values(sort_col).iloc[0] if sort_col else metrics.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Best model", str(best_row.get("model", best_row.get("model_name", "Best model"))), "Selected by lowest MAE", "blue-card")
        with c2: metric_card("MAE", fmt_eur(best_row.get("test_mae", best_row.get("mae", np.nan))), "Average absolute error")
        with c3: metric_card("RMSE", fmt_eur(best_row.get("test_rmse", best_row.get("rmse", np.nan))), "Large-error sensitive")
        with c4: metric_card("R²", fmt_num(best_row.get("test_r2", best_row.get("r2", np.nan)), decimals=2), "Explained variance")
        st.dataframe(metrics, hide_index=True, width="stretch")
        if "model" in metrics.columns and "test_mae" in metrics.columns:
            show_bar(metrics.sort_values("test_mae").set_index("model")["test_mae"], "Test MAE by Model", "MAE (€)")
        elif "model_name" in metrics.columns and "mae" in metrics.columns:
            show_bar(metrics.sort_values("mae").set_index("model_name")["mae"], "MAE by Model", "MAE (€)")
        if predictions_df is not None and {"actual_price", "predicted_price"}.issubset(predictions_df.columns):
            show_scatter(predictions_df, "actual_price", "predicted_price", "Actual vs Predicted Property Prices")

    elif page == "Opportunity Explorer":
        hero("Opportunity Explorer", "Rank properties by model-estimated upside.", "Investment Scoring")
        if predictions_df is None:
            st.error("No predictions file found.")
            return
        opp = predictions_df.copy()
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Predictions", fmt_num(len(opp)), "Test-set properties")
        with c2: metric_card("Top score", fmt_num(opp["opportunity_score"].max(), decimals=0), "Best opportunity", "blue-card")
        with c3: metric_card("Avg. gap", fmt_num(opp["percentage_error"].mean(), "%", 1), "Predicted vs actual")
        with c4: metric_card("Opportunities", fmt_num((opp["valuation_status"] == "Potentially undervalued").sum()), "Predicted upside > 10%")
        f1, f2, f3 = st.columns(3)
        min_score = f1.slider("Minimum score", 0, 100, 55)
        max_price = f2.number_input("Maximum actual price (€)", min_value=0, value=int(opp["actual_price"].max()), step=100_000)
        selected_district = f3.selectbox("District", ["All"] + sorted(opp["district"].dropna().astype(str).unique())) if "district" in opp.columns else "All"
        filtered = opp[(opp["opportunity_score"] >= min_score) & (opp["actual_price"] <= max_price)]
        if selected_district != "All" and "district" in filtered.columns:
            filtered = filtered[filtered["district"].astype(str) == selected_district]
        filtered = filtered.sort_values("opportunity_score", ascending=False)
        cols = [c for c in ["opportunity_score", "valuation_status", "actual_price", "predicted_price", "percentage_error", "district", "neighborhood", "propertyType", "size", "rooms", "bathrooms"] if c in filtered.columns]
        st.dataframe(filtered[cols].head(30), hide_index=True, width="stretch")

    elif page == "Manual Estimator":
        hero("Manual Fair Price Estimator", "Estimate a fair market price from user inputs.", "Interactive Prediction")
        if model_df is None:
            st.error("No model-ready dataset found.")
            return
        model = None
        model_path = paths.get("model")
        if model_path is not None and model_path.exists():
            try: model = load_pickle_model(str(model_path))
            except Exception as exc: st.warning(f"Model could not be loaded, fallback estimator used: {exc}")
        c1, c2, c3 = st.columns(3)
        size = c1.number_input("Size (m²)", min_value=20, max_value=1000, value=120, step=5)
        rooms = c2.number_input("Rooms", min_value=0, max_value=10, value=3, step=1)
        bathrooms = c3.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
        c4, c5, c6 = st.columns(3)
        district = c4.selectbox("District", sorted(model_df["district"].dropna().astype(str).unique()) if "district" in model_df.columns else ["Unknown"])
        property_type = c5.selectbox("Property type", sorted(model_df["propertyType"].dropna().astype(str).unique()) if "propertyType" in model_df.columns else ["flat"])
        asking_price = c6.number_input("Actual listing price (€)", min_value=0, value=1_000_000, step=50_000)
        if st.button("Estimate fair price"):
            district_rows = model_df[model_df["district"].astype(str) == district] if "district" in model_df.columns else model_df
            predicted_price = float(((district_rows["price"] / district_rows["size"]).median()) * size)
            gap_pct = (predicted_price - asking_price) / asking_price * 100 if asking_price else np.nan
            o1, o2, o3 = st.columns(3)
            with o1: metric_card("Estimated fair price", fmt_eur(predicted_price), "District median €/m² fallback")
            with o2: metric_card("Actual listing price", fmt_eur(asking_price), "User input")
            with o3: metric_card("Gap", fmt_num(gap_pct, "%", 1), "Predicted vs listed", "blue-card")

    elif page == "Data Health":
        hero("Data Health", "Inspect missing values, column types and dataset quality.", "Technical Quality Report")
        available = [name for name in ["raw", "model_ready", "predictions", "metrics", "main_metrics"] if data.get(name) is not None]
        if not available:
            st.error("No datasets available.")
            return
        selected = st.selectbox("Dataset", available)
        selected_df = data[selected]
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Rows", fmt_num(len(selected_df)), "Dataset size")
        with c2: metric_card("Columns", fmt_num(selected_df.shape[1]), "Variables")
        with c3: metric_card("Missing values", fmt_num(int(selected_df.isna().sum().sum())), "Total missing cells")
        with c4: metric_card("Duplicates", fmt_num(int(selected_df.duplicated().sum())), "Exact duplicates", "blue-card")
        health = pd.DataFrame({"column": selected_df.columns, "dtype": selected_df.dtypes.astype(str).values, "missing_count": selected_df.isna().sum().values, "missing_pct": (selected_df.isna().mean().values * 100).round(2), "unique_values": selected_df.nunique(dropna=True).values}).sort_values("missing_pct", ascending=False)
        st.dataframe(health, hide_index=True, width="stretch")
        st.subheader("Preview")
        st.dataframe(selected_df.head(80), hide_index=True, width="stretch")


if __name__ == "__main__":
    build_app()
