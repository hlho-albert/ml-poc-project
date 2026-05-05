"""
src/data.py

Reusable data loading, cleaning and feature engineering utilities for the
Idealista Investment Opportunity Detector project.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def find_latest_file(directory: str | Path, pattern: str) -> Path:
    """Return the latest file matching a pattern inside a directory."""
    directory = Path(directory)
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file matching {pattern!r} found in {directory}")
    return files[-1]


def load_latest_dataset(project_root: str | Path = ".") -> pd.DataFrame:
    """
    Load the latest available Idealista dataset.

    Priority:
    1. data/processed/idealista_eda_enriched_*.csv
    2. data/raw/idealista_raw_listings_*.csv
    """
    project_root = Path(project_root)
    processed_dir = project_root / "data" / "processed"
    raw_dir = project_root / "data" / "raw"

    candidates = []
    candidates.extend(sorted(processed_dir.glob("idealista_eda_enriched_*.csv")))
    candidates.extend(sorted(raw_dir.glob("idealista_raw_listings_*.csv")))

    if not candidates:
        raise FileNotFoundError(
            "No Idealista dataset found. Run data collection and EDA notebooks first."
        )

    return pd.read_csv(candidates[-1])


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names for easier use in modeling code."""
    out = df.copy()
    out.columns = [
        col.strip()
           .replace(".", "_")
           .replace(" ", "_")
           .replace("-", "_")
        for col in out.columns
    ]
    return out


def to_bool_series(series: pd.Series) -> pd.Series:
    """Convert mixed boolean/string/numeric values into booleans with NaN support."""
    true_values = {"true", "yes", "y", "1", "si", "sí"}
    false_values = {"false", "no", "n", "0"}

    def convert(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not pd.isna(value):
            if value == 1:
                return True
            if value == 0:
                return False
        value_str = str(value).strip().lower()
        if value_str in true_values:
            return True
        if value_str in false_values:
            return False
        return np.nan

    return series.apply(convert)


def parse_floor_value(value) -> float:
    """
    Convert Idealista floor values into approximate numerical values.
    Examples:
    - bj / bajo / pb -> 0
    - ss / sotano / semisotano -> -1
    - en / entreplanta -> 0.5
    """
    if pd.isna(value):
        return np.nan

    value_str = str(value).strip().lower()

    if value_str in {"bj", "bajo", "pb", "principal"}:
        return 0.0
    if value_str in {"ss", "sotano", "sótano", "semisotano", "semisótano"}:
        return -1.0
    if value_str in {"en", "entreplanta"}:
        return 0.5

    match = re.search(r"-?\d+", value_str)
    if match:
        return float(match.group(0))

    return np.nan


def safe_divide(a, b):
    """Vectorized safe division."""
    return np.where((b != 0) & (~pd.isna(b)), a / b, np.nan)


def find_first_existing_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Return the first column that exists in a DataFrame."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic cleaning: column names, duplicates, data types."""
    out = normalize_column_names(df)

    if "propertyCode" in out.columns:
        out = out.drop_duplicates(subset=["propertyCode"], keep="first")

    numeric_cols = [
        "price", "size", "rooms", "bathrooms", "latitude", "longitude",
        "distance", "numPhotos", "price_per_m2"
    ]

    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    boolean_source_cols = [
        "hasLift",
        "exterior",
        "parkingSpace_hasParkingSpace",
        "parkingSpace_isParkingSpaceIncludedInPrice",
        "newDevelopmentFinished",
        "showAddress",
    ]

    for col in boolean_source_cols:
        if col in out.columns:
            out[col] = to_bool_series(out[col])

    if "floor" in out.columns:
        out["floor_numeric"] = out["floor"].apply(parse_floor_value)
        out["floor_missing"] = out["floor_numeric"].isna().astype(int)

    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create real estate features for analysis and modeling."""
    out = df.copy()

    bool_mapping = {
        "hasLift": "has_lift",
        "exterior": "is_exterior",
        "parkingSpace_hasParkingSpace": "has_parking",
        "parkingSpace_isParkingSpaceIncludedInPrice": "parking_included",
        "newDevelopmentFinished": "new_development_finished",
    }

    for source_col, new_col in bool_mapping.items():
        if source_col in out.columns:
            out[f"{new_col}_missing"] = out[source_col].isna().astype(int)
            out[new_col] = out[source_col].fillna(False).astype(bool).astype(int)

    if "price" in out.columns and "size" in out.columns:
        out["price_per_m2"] = safe_divide(out["price"], out["size"])

    if "rooms" in out.columns and "size" in out.columns:
        out["rooms_per_100m2"] = safe_divide(out["rooms"], out["size"]) * 100

    if "bathrooms" in out.columns and "rooms" in out.columns:
        out["bathrooms_per_room"] = safe_divide(
            out["bathrooms"], out["rooms"].replace(0, np.nan)
        )

    if "size" in out.columns:
        out["log_size"] = np.log1p(out["size"])

    if "price" in out.columns:
        out["log_price"] = np.log1p(out["price"])

    if "numPhotos" in out.columns:
        out["log_num_photos"] = np.log1p(out["numPhotos"])

    if "district" in out.columns and "price_per_m2" in out.columns:
        out["district_median_price_per_m2"] = (
            out.groupby("district")["price_per_m2"].transform("median")
        )
        out["district_listing_count"] = (
            out.groupby("district")["price_per_m2"].transform("count")
        )
        out["discount_vs_district_median_pct"] = (
            (out["price_per_m2"] - out["district_median_price_per_m2"])
            / out["district_median_price_per_m2"]
            * 100
        )

    description_col = find_first_existing_col(
        out,
        ["description", "propertyComment", "suggestedTexts_subtitle", "suggestedTexts_title"],
    )

    if description_col:
        text = out[description_col].fillna("").astype(str).str.lower()
        out["description_length"] = text.str.len()

        luxury_keywords = [
            "lujo", "exclusivo", "premium", "alto standing", "vistas",
            "ático", "piscina", "spa", "domótica", "diseño",
        ]
        renovation_keywords = [
            "reformado", "a reformar", "reforma", "renovado", "rehabilitado",
        ]
        investment_keywords = [
            "oportunidad", "inversión", "rentabilidad", "inversor",
        ]

        out["has_luxury_keywords"] = text.apply(
            lambda s: int(any(keyword in s for keyword in luxury_keywords))
        )
        out["has_renovation_keywords"] = text.apply(
            lambda s: int(any(keyword in s for keyword in renovation_keywords))
        )
        out["has_investment_keywords"] = text.apply(
            lambda s: int(any(keyword in s for keyword in investment_keywords))
        )
    else:
        out["description_length"] = 0
        out["has_luxury_keywords"] = 0
        out["has_renovation_keywords"] = 0
        out["has_investment_keywords"] = 0

    out["has_price_drop"] = 0
    price_drop_cols = [
        "priceInfo_price_priceDropInfo_formerPrice",
        "priceInfo_price_priceDropInfo_priceDropValue",
        "priceInfo_price_priceDropInfo_priceDropPercentage",
    ]
    available_price_drop_cols = [col for col in price_drop_cols if col in out.columns]
    if available_price_drop_cols:
        out["has_price_drop"] = out[available_price_drop_cols].notna().any(axis=1).astype(int)

    return out


def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Apply broad sanity filters for the premium Madrid dataset."""
    out = df.copy()
    filters = pd.Series(True, index=out.index)

    if "price" in out.columns:
        filters &= out["price"].between(100_000, 10_000_000)

    if "size" in out.columns:
        filters &= out["size"].between(20, 1_000)

    if "rooms" in out.columns:
        filters &= out["rooms"].between(0, 10)

    if "bathrooms" in out.columns:
        filters &= out["bathrooms"].between(1, 10)

    if "price_per_m2" in out.columns:
        filters &= out["price_per_m2"].between(1_000, 30_000)

    return out[filters].copy()


def build_model_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a model-ready dataset for predicting `price`.

    Target-derived variables such as `price_per_m2` and district median price per m²
    are intentionally excluded from model inputs to reduce target leakage.
    """
    out = df.copy()

    target_col = "price"

    candidate_feature_cols = [
        "size", "rooms", "bathrooms", "latitude", "longitude",
        "distance", "numPhotos", "floor_numeric",
        "rooms_per_100m2", "bathrooms_per_room",
        "log_size", "log_num_photos",
        "description_length",
        "has_lift", "has_lift_missing",
        "is_exterior", "is_exterior_missing",
        "has_parking", "has_parking_missing",
        "parking_included", "parking_included_missing",
        "new_development_finished", "new_development_finished_missing",
        "has_price_drop",
        "has_luxury_keywords",
        "has_renovation_keywords",
        "has_investment_keywords",
        "floor_missing",
        "propertyType",
        "district",
        "neighborhood",
        "municipality",
        "detailedType_typology",
        "detailedType_subTypology",
    ]

    feature_cols = [col for col in candidate_feature_cols if col in out.columns]
    model_df = out[[target_col] + feature_cols].copy()

    essential_cols = [col for col in ["price", "size", "rooms", "bathrooms"] if col in model_df.columns]
    model_df = model_df.dropna(subset=essential_cols)

    num_features = [col for col in model_df.select_dtypes(include=[np.number]).columns if col != target_col]
    cat_features = model_df.select_dtypes(exclude=[np.number]).columns.tolist()

    for col in num_features:
        model_df[col] = model_df[col].fillna(model_df[col].median())

    for col in cat_features:
        model_df[col] = model_df[col].fillna("Unknown").astype(str)

    return model_df


def prepare_model_ready_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning and feature engineering pipeline."""
    cleaned = basic_cleaning(df)
    featured = engineer_features(cleaned)
    filtered = filter_outliers(featured)
    model_df = build_model_dataset(filtered)
    return model_df
