import csv
import os
import threading
from datetime import datetime, timezone

from src.config import PREDICTION_LOG_PATH
from src.features import FEATURES

LOG_COLUMNS = ["timestamp", "model_version", "date"] + list(FEATURES) + ["prediction"]

_lock = threading.Lock()


def log_path() -> str:
    return os.getenv("PREDICTION_LOG_PATH", PREDICTION_LOG_PATH)


def log_prediction(input_data: dict, prediction: float, model_version: str):
    """Journalise chaque prédiction : c'est la matière première du suivi de dérive."""
    path = log_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": model_version,
        "date": input_data.get("date") or "",
        "prediction": prediction,
    }
    row.update({feature: input_data[feature] for feature in FEATURES})

    with _lock:
        new_file = not os.path.exists(path)

        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)

            if new_file:
                writer.writeheader()

            writer.writerow(row)
