"""
src/train.py

Training script for the Idealista Investment Opportunity Detector project.
This script mirrors the logic from notebooks/04_model_training.ipynb.
"""

from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

RANDOM_STATE = 42


def main(project_root: str | Path = "."):
    project_root = Path(project_root)
    processed_dir = project_root / "data" / "processed"
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    data_path = processed_dir / "idealista_model_ready_latest.csv"
    df = pd.read_csv(data_path)

    target_col = "price"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }

    pipelines = {
        name: Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        for name, model in models.items()
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    results = []
    trained = {}

    for name, pipeline in pipelines.items():
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        pipeline.fit(X_train, y_train)
        trained[name] = pipeline
        y_pred = pipeline.predict(X_test)

        results.append({
            "model": name,
            "cv_mae_mean": -scores["test_mae"].mean(),
            "cv_rmse_mean": -scores["test_rmse"].mean(),
            "cv_r2_mean": scores["test_r2"].mean(),
            "test_mae": mean_absolute_error(y_test, y_pred),
            "test_rmse": mean_squared_error(y_test, y_pred, squared=False),
            "test_r2": r2_score(y_test, y_pred),
        })

    metrics = pd.DataFrame(results).sort_values("test_mae")
    best_model_name = metrics.iloc[0]["model"]
    best_model = trained[best_model_name]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics.to_csv(reports_dir / f"model_metrics_{timestamp}.csv", index=False)
    metrics.to_csv(reports_dir / "model_metrics_latest.csv", index=False)

    joblib.dump(best_model, models_dir / f"best_model_{timestamp}.pkl")
    joblib.dump(best_model, models_dir / "best_model_latest.pkl")

    metadata = {
        "best_model": best_model_name,
        "target_col": target_col,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "models": list(models.keys()),
    }

    with open(reports_dir / "model_training_metadata_latest.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(metrics)
    print(f"Best model: {best_model_name}")


if __name__ == "__main__":
    main()
