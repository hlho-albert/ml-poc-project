from __future__ import annotations

import pandas as pd

try:
    import config
except ImportError:
    from . import config  # type: ignore


def write_metrics(rows: list[dict[str, object]]) -> pd.DataFrame:
    """
    Write model evaluation metrics to disk.

    Main output expected by the professor's main.py:
        results/model_metrics.csv

    Additional copy:
        reports/model_metrics_from_main.csv
    """
    if not rows:
        raise ValueError("No metric rows were provided.")

    df = pd.DataFrame(rows)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results_path = config.RESULTS_DIR / "model_metrics.csv"
    reports_path = config.REPORTS_DIR / "model_metrics_from_main.csv"

    df.to_csv(results_path, index=False, encoding="utf-8-sig")
    df.to_csv(reports_path, index=False, encoding="utf-8-sig")

    return df
