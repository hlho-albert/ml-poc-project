from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_DIR = PROJECT_ROOT / "results"

ENV_FILE = PROJECT_ROOT / ".env"
APP_ENTRYPOINT =  SRC_DIR / "app.py"

STREAMLIT_HOST = os.getenv("STREAMLIT_HOST", "localhost")
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

MODELS = {
    "best_model": {
        "name": "Gradient Boosting Regressor",
        "path": MODELS_DIR / "best_model_latest.pkl",
    }
}

MODEL_READY_DATASET = PROCESSED_DATA_DIR / "idealista_model_ready_latest.csv"
TARGET_COLUMN = "price"
RANDOM_STATE = 42
TEST_SIZE = 0.20
