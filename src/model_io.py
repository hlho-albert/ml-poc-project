from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def load_model(path: Path) -> Any:
    """
    Load a trained model or sklearn Pipeline from disk.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. "
            "Run notebooks/04_model_training.ipynb first."
        )

    return joblib.load(path)


def save_model(model: Any, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
