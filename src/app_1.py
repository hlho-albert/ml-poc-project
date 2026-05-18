"""
app.py — Idealista Investment Opportunity Detector

Run from the project root with:

    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

st.set_page_config(
    page_title="Idealista Investment Detector",
    page_icon="🏠",
    layout="wide",
)


def find_latest_file(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern))
    if not files:
        return None
    return files[-1]


@st.cache_data
def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def get_main_paths():
    return {
        "raw": find_latest_file(RAW_DATA_DIR, "idealista_raw_listings_*.csv"),
        "clean": find_latest_file(PROCESSED_DATA_DIR, "idealista_clean_full_*.csv"),
        "model_ready": PROCESSED_DATA_DIR / "idealista_model_ready_latest.csv",
        "predictions": PROCESSED_DATA_DIR / "model_predictions_latest.csv",
        "metrics": REPORTS_DIR / "model_metrics_latest.csv",
        "model": MODELS_DIR / "best_model_latest.pkl",
    }


def load_available_data(paths: dict[str, Path | None]):
    data = {}
    for name, path in paths.items():
        if name == "model":
            continue
        if path is not None and Path(path).exists():
            try:
                data[name] = load_csv(path)
            except Exception as exc:
                st.warning(f"Could not load {path}: {exc}")
    return data


def simple_opportunity_score(row: pd.Series) -> float:
    if "predicted_price" not in row or "actual_price" not in row:
        return np.nan
    actual = row["actual_price"]
    predicted = row["predicted_price"]
    if pd.isna(actual) or actual <= 0 or pd.isna(predicted):
        return np.nan
    gap_pct = (predicted - actual) / actual * 100
    score = 50 + gap_pct * 2.5
    return float(np.clip(score, 0, 100))


paths = get_main_paths()
data = load_available_data(paths)
raw_df = data.get("raw")
clean_df = data.get("clean")
model_df = data.get("model_ready")
predictions_df = data.get("predictions")
metrics_df = data.get("metrics")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a section",
    [
        "Project Overview",
        "Raw Data",
        "Model Performance",
        "Opportunity Explorer",
        "Manual Estimator",
        "Data Health",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("Idealista Investment Opportunity Detector")

st.title("🏠 Idealista Investment Opportunity Detector")
st.caption("ML project for Madrid premium real estate opportunity detection")


if page == "Project Overview":
    st.header("Project Overview")
    st.markdown(
        """
        This application uses property listing data collected from the Idealista API
        to estimate real estate prices and identify potentially undervalued properties.

        The core business question is:

        **Is this property priced below, above, or in line with its estimated fair market value?**
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    if model_df is not None:
        col1.metric("Model-ready rows", f"{len(model_df):,}")
        col2.metric("Features", f"{max(model_df.shape[1] - 1, 0):,}")
        col3.metric("Median price", f"€{model_df['price'].median():,.0f}" if "price" in model_df.columns else "N/A")
        col4.metric("Median size", f"{model_df['size'].median():,.0f} m²" if "size" in model_df.columns else "N/A")
    else:
        col1.metric("Model-ready rows", "N/A")
        col2.metric("Features", "N/A")
        col3.metric("Median price", "N/A")
        col4.metric("Median size", "N/A")

    st.subheader("Pipeline")
    st.markdown(
        """
        1. **Data collection** from the Idealista API  
        2. **EDA** of prices, districts and property characteristics  
        3. **Data cleaning and feature engineering**  
        4. **Model training** with Ridge, Random Forest and Gradient Boosting  
        5. **Opportunity scoring** based on predicted vs actual listing price  
        6. **Streamlit interface** for exploration and demonstration  
        """
    )

    st.subheader("Available files")
    file_status = pd.DataFrame(
        [[name, str(path), bool(path is not None and Path(path).exists())] for name, path in paths.items()],
        columns=["File", "Path", "Available"],
    )
    st.dataframe(file_status, use_container_width=True)


elif page == "Raw Data":
    st.header("Raw Data Exploration")
    df_display = raw_df if raw_df is not None else model_df
    if df_display is None:
        st.error("No dataset found. Run the notebooks first.")
    else:
        st.subheader("Dataset preview")
        st.dataframe(df_display.head(50), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{len(df_display):,}")
        col2.metric("Columns", f"{df_display.shape[1]:,}")
        if "price" in df_display.columns:
            prices = pd.to_numeric(df_display["price"], errors="coerce")
            col3.metric("Median price", f"€{prices.median():,.0f}")
            st.subheader("Raw price distribution")
            st.bar_chart(prices.dropna().value_counts(bins=30).sort_index())

        if {"district", "price"}.issubset(df_display.columns):
            st.subheader("Median price by district")
            temp = df_display.copy()
            temp["price"] = pd.to_numeric(temp["price"], errors="coerce")
            district_price = temp.dropna(subset=["price"]).groupby("district")["price"].median().sort_values(ascending=False).head(20)
            st.bar_chart(district_price)


elif page == "Model Performance":
    st.header("Model Performance")
    if metrics_df is None:
        st.error("No model metrics file found. Run notebooks/04_model_training.ipynb first.")
    else:
        st.subheader("Metrics table")
        st.dataframe(metrics_df, use_container_width=True)
        available_metrics = [c for c in ["test_mae", "test_rmse", "test_r2", "cv_mae_mean", "cv_rmse_mean", "cv_r2_mean"] if c in metrics_df.columns]
        metric_choice = st.selectbox("Choose metric to compare", available_metrics)
        if metric_choice:
            st.bar_chart(metrics_df.set_index("model")[metric_choice])

        if predictions_df is not None and {"actual_price", "predicted_price"}.issubset(predictions_df.columns):
            st.subheader("Actual vs Predicted Prices")
            sample_size = st.slider("Number of predictions to display", 10, min(200, len(predictions_df)), min(100, len(predictions_df)), 10)
            st.scatter_chart(predictions_df[["actual_price", "predicted_price"]].head(sample_size), x="actual_price", y="predicted_price")
            mae = np.mean(np.abs(predictions_df["predicted_price"] - predictions_df["actual_price"]))
            st.metric("Average absolute error", f"€{mae:,.0f}")


elif page == "Opportunity Explorer":
    st.header("Opportunity Explorer")
    if predictions_df is None:
        st.error("No predictions file found. Run notebooks/04_model_training.ipynb first.")
    else:
        df_opp = predictions_df.copy()
        if "opportunity_score" not in df_opp.columns:
            df_opp["opportunity_score"] = df_opp.apply(simple_opportunity_score, axis=1)
        if "percentage_error" not in df_opp.columns and {"predicted_price", "actual_price"}.issubset(df_opp.columns):
            df_opp["percentage_error"] = (df_opp["predicted_price"] - df_opp["actual_price"]) / df_opp["actual_price"] * 100

        st.markdown("Higher score = stronger statistical underpricing according to the model.")
        col1, col2, col3 = st.columns(3)
        min_score = col1.slider("Minimum opportunity score", 0, 100, 60)
        max_actual_price = col2.number_input("Maximum actual price (€)", min_value=0, value=int(df_opp["actual_price"].max()), step=100_000)
        if "district" in df_opp.columns:
            districts = ["All"] + sorted(df_opp["district"].dropna().astype(str).unique().tolist())
            selected_district = col3.selectbox("District", districts)
        else:
            selected_district = "All"

        filtered = df_opp[df_opp["opportunity_score"] >= min_score]
        filtered = filtered[filtered["actual_price"] <= max_actual_price]
        if selected_district != "All" and "district" in filtered.columns:
            filtered = filtered[filtered["district"].astype(str) == selected_district]
        filtered = filtered.sort_values("opportunity_score", ascending=False)

        display_cols = ["opportunity_score", "actual_price", "predicted_price", "percentage_error", "district", "neighborhood", "propertyType", "size", "rooms", "bathrooms"]
        display_cols = [c for c in display_cols if c in filtered.columns]
        st.dataframe(filtered[display_cols].head(30), use_container_width=True)

        if len(filtered) > 0:
            selected_index = st.selectbox("Select row index", filtered.index.tolist())
            selected = filtered.loc[selected_index]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Actual price", f"€{selected['actual_price']:,.0f}")
            c2.metric("Predicted price", f"€{selected['predicted_price']:,.0f}")
            c3.metric("Gap", f"{selected.get('percentage_error', np.nan):,.1f}%")
            c4.metric("Score", f"{selected.get('opportunity_score', np.nan):,.0f}/100")
            st.json(selected.to_dict())


elif page == "Manual Estimator":
    st.header("Manual Estimator")
    model_path = paths.get("model")
    trained_model = None
    if model_path is not None and model_path.exists():
        try:
            import joblib
            trained_model = joblib.load(model_path)
        except Exception as exc:
            st.warning(f"Could not load model. Mock fallback will be used. Error: {exc}")
    else:
        st.warning("No trained model found. Mock fallback will be used.")

    if model_df is None:
        st.error("No model-ready dataset found.")
    else:
        col1, col2, col3 = st.columns(3)
        size = col1.number_input("Size (m²)", min_value=20, max_value=1000, value=120, step=5)
        rooms = col2.number_input("Rooms", min_value=0, max_value=10, value=3, step=1)
        bathrooms = col3.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)

        col4, col5, col6 = st.columns(3)
        district = col4.selectbox("District", sorted(model_df["district"].dropna().astype(str).unique())) if "district" in model_df.columns else col4.text_input("District", "Barrio de Salamanca")
        property_type = col5.selectbox("Property type", sorted(model_df["propertyType"].dropna().astype(str).unique())) if "propertyType" in model_df.columns else col5.text_input("Property type", "flat")
        asking_price = col6.number_input("Actual listing price (€)", min_value=0, value=1_000_000, step=50_000)

        col7, col8, col9 = st.columns(3)
        has_lift = col7.checkbox("Has lift", value=True)
        has_parking = col8.checkbox("Has parking", value=False)
        is_exterior = col9.checkbox("Exterior", value=True)

        defaults = {}
        for col in model_df.drop(columns=["price"], errors="ignore").columns:
            if pd.api.types.is_numeric_dtype(model_df[col]):
                defaults[col] = model_df[col].median()
            else:
                mode = model_df[col].mode()
                defaults[col] = mode.iloc[0] if len(mode) else "Unknown"
        input_row = pd.DataFrame([defaults])

        overrides = {
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
        for col, val in overrides.items():
            if col in input_row.columns:
                input_row[col] = val

        if st.button("Estimate fair price"):
            if trained_model is not None:
                try:
                    predicted_price = float(trained_model.predict(input_row)[0])
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    predicted_price = np.nan
            else:
                predicted_price = (model_df["price"].median() / model_df["size"].median()) * size

            if not pd.isna(predicted_price):
                gap_pct = (predicted_price - asking_price) / asking_price * 100 if asking_price else np.nan
                score = float(np.clip(50 + gap_pct * 2.5, 0, 100)) if not pd.isna(gap_pct) else np.nan
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Estimated fair price", f"€{predicted_price:,.0f}")
                c2.metric("Actual listing price", f"€{asking_price:,.0f}")
                c3.metric("Gap", f"{gap_pct:,.1f}%")
                c4.metric("Opportunity score", f"{score:,.0f}/100")
                if gap_pct >= 10:
                    st.success("The property may be potentially undervalued.")
                elif gap_pct <= -10:
                    st.error("The property may be potentially overvalued.")
                else:
                    st.info("The property appears fairly priced.")


elif page == "Data Health":
    st.header("Data Health")
    available = [name for name, df_ in data.items() if df_ is not None]
    if not available:
        st.error("No datasets available.")
    else:
        selected_name = st.selectbox("Select dataset", available)
        selected_df = data[selected_name]
        st.write("Shape:", selected_df.shape)
        health = pd.DataFrame({
            "column": selected_df.columns,
            "dtype": selected_df.dtypes.astype(str).values,
            "missing_count": selected_df.isna().sum().values,
            "missing_pct": (selected_df.isna().mean().values * 100).round(2),
            "unique_values": selected_df.nunique(dropna=True).values,
        }).sort_values("missing_pct", ascending=False)
        st.dataframe(health, use_container_width=True)
        st.subheader("Preview")
        st.dataframe(selected_df.head(50), use_container_width=True)
