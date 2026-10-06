import os
import sys
import json
from datetime import datetime, timezone

import pandas as pd
import mlflow

from dotenv import load_dotenv
from mlflow import MlflowClient
from evidently import Report, Dataset, DataDefinition, Regression
from evidently.presets import DataDriftPreset, RegressionPreset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import FEATURES, TARGET
from src.config import (
    TRAIN_RATIO,
    PROCESSED_DATA_PATH,
    METRICS_PATH,
    PREDICTION_LOG_PATH,
    MONITORING_SUMMARY_PATH,
    MONITORING_REPORTS_DIR,
    MONITORING_WINDOW,
    MONITORING_MIN_ROWS,
    DRIFT_SHARE_THRESHOLD,
    PERFORMANCE_DEGRADATION_FACTOR,
    MLFLOW_MONITORING_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    ALIAS_PRODUCTION,
)

load_dotenv()

DRIFT_REPORT_PATH = os.path.join(MONITORING_REPORTS_DIR, "data_drift_report.html")
PERFORMANCE_REPORT_PATH = os.path.join(MONITORING_REPORTS_DIR, "performance_report.html")


def load_prediction_log() -> pd.DataFrame:
    path = os.getenv("PREDICTION_LOG_PATH", PREDICTION_LOG_PATH)

    if not os.path.exists(path):
        return pd.DataFrame()

    return pd.read_csv(path).tail(MONITORING_WINDOW)


def reference_mae() -> float:
    """MAE du modèle en production, telle que mesurée à l'entraînement."""
    try:
        version = MlflowClient().get_model_version_by_alias(
            MLFLOW_MODEL_NAME, ALIAS_PRODUCTION
        )
        return float(version.tags["mae"])
    except Exception:
        with open(METRICS_PATH, encoding="utf-8") as f:
            return float(json.load(f)["mae"])


def is_drifted(method: str, score: float, threshold: float) -> bool:
    # Tests statistiques : dérive si la p-value passe sous le seuil.
    # Distances (Wasserstein, Jensen-Shannon, PSI) : dérive si elles le dépassent.
    if "p_value" in method:
        return score < threshold

    return score > threshold


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Compare les entrées reçues en production aux données d'entraînement."""
    definition = DataDefinition(numerical_columns=list(FEATURES))

    snapshot = Report([DataDriftPreset(drift_share=DRIFT_SHARE_THRESHOLD)]).run(
        Dataset.from_pandas(current[FEATURES], data_definition=definition),
        Dataset.from_pandas(reference[FEATURES], data_definition=definition),
    )
    snapshot.save_html(DRIFT_REPORT_PATH)

    drift_share = 0.0
    drift_scores = {}
    drifted_features = []

    for metric in snapshot.dict()["metrics"]:
        config = metric["config"]

        if config["type"].endswith("DriftedColumnsCount"):
            drift_share = float(metric["value"]["share"])
        elif config["type"].endswith("ValueDrift"):
            score = float(metric["value"])
            drift_scores[config["column"]] = round(score, 4)

            if is_drifted(config["method"], score, config["threshold"]):
                drifted_features.append(config["column"])

    return {
        "drift_share": drift_share,
        "drifted_features_count": len(drifted_features),
        "drifted_features": drifted_features,
        "drift_scores": drift_scores,
        "drift_detected": int(drift_share >= DRIFT_SHARE_THRESHOLD),
    }


def run_performance_report(current: pd.DataFrame, actuals: pd.DataFrame):
    """Rapproche les prédictions servies des valeurs réellement observées à J+7.
    Seules les requêtes datées, dont la valeur réelle est déjà connue, comptent."""
    if "date" not in current.columns:
        return None

    dated = current.dropna(subset=["date"])
    scored = dated.merge(actuals[["date", TARGET]], on="date", how="inner")

    if len(scored) < MONITORING_MIN_ROWS:
        return None

    definition = DataDefinition(
        numerical_columns=list(FEATURES),
        regression=[Regression(target=TARGET, prediction="prediction")],
    )

    snapshot = Report([RegressionPreset()]).run(
        Dataset.from_pandas(
            scored[list(FEATURES) + [TARGET, "prediction"]],
            data_definition=definition,
        )
    )
    snapshot.save_html(PERFORMANCE_REPORT_PATH)

    for metric in snapshot.dict()["metrics"]:
        if metric["config"]["type"].endswith(":MAE"):
            return {
                "production_mae": float(metric["value"]["mean"]),
                "production_mean_actual": float(scored[TARGET].mean()),
                "scored_rows": len(scored),
            }

    return None


def relative_mae(mae: float, mean_actual: float):
    """MAE rapportée au niveau moyen des cas : la MAE brute dépend de l'ampleur
    de l'épidémie, ce rapport reste comparable d'une période à l'autre."""
    if not mean_actual:
        return None

    return mae / mean_actual


def monitor_model():
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    os.makedirs(MONITORING_REPORTS_DIR, exist_ok=True)

    now = datetime.now(timezone.utc)

    summary = {
        "timestamp": now.isoformat(timespec="seconds"),
        "timestamp_unix": int(now.timestamp()),
        "status": "ok",
        "analysed_rows": 0,
        "model_version": None,
        "drift_share": None,
        "drifted_features_count": None,
        "drifted_features": [],
        "drift_scores": {},
        "drift_detected": 0,
        "production_mae": None,
        "reference_mae": None,
        "production_relative_mae": None,
        "reference_relative_mae": None,
        "scored_rows": 0,
        "performance_degraded": 0,
        "retrain_recommended": False,
    }

    current = load_prediction_log()
    summary["analysed_rows"] = int(len(current))

    if len(current) < MONITORING_MIN_ROWS:
        summary["status"] = "donnees_insuffisantes"
        print(
            f"{len(current)} prédictions journalisées, {MONITORING_MIN_ROWS} requises : "
            "pas de verdict de dérive."
        )
    else:
        df_model = pd.read_csv(PROCESSED_DATA_PATH)
        train_size = int(len(df_model) * TRAIN_RATIO)
        reference = df_model.iloc[:train_size]

        summary["model_version"] = str(current["model_version"].iloc[-1])
        summary.update(run_drift_report(reference, current))

        performance = run_performance_report(current, df_model)

        if performance is not None:
            production_mean_actual = performance.pop("production_mean_actual")
            summary.update(performance)
            summary["reference_mae"] = reference_mae()

            # Référence : la MAE d'entraînement, mesurée sur le jeu de test
            summary["production_relative_mae"] = relative_mae(
                summary["production_mae"], production_mean_actual
            )
            summary["reference_relative_mae"] = relative_mae(
                summary["reference_mae"], float(df_model[TARGET].iloc[train_size:].mean())
            )

            if summary["production_relative_mae"] and summary["reference_relative_mae"]:
                summary["performance_degraded"] = int(
                    summary["production_relative_mae"]
                    > PERFORMANCE_DEGRADATION_FACTOR * summary["reference_relative_mae"]
                )

        summary["retrain_recommended"] = bool(
            summary["drift_detected"] or summary["performance_degraded"]
        )

    with open(MONITORING_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # L'historique des contrôles est conservé dans MLflow, rapports HTML compris
    mlflow.set_experiment(MLFLOW_MONITORING_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"monitoring_{now:%Y%m%d_%H%M%S}"):
        mlflow.set_tag("status", summary["status"])
        mlflow.set_tag("model_version", summary["model_version"])
        mlflow.set_tag("drifted_features", ", ".join(summary["drifted_features"]))

        for key in (
            "analysed_rows",
            "drift_share",
            "drifted_features_count",
            "drift_detected",
            "production_mae",
            "reference_mae",
            "production_relative_mae",
            "reference_relative_mae",
            "scored_rows",
            "performance_degraded",
        ):
            if summary[key] is not None:
                mlflow.log_metric(key, float(summary[key]))

        mlflow.log_artifact(MONITORING_SUMMARY_PATH)

        if summary["status"] == "ok":
            mlflow.log_artifact(DRIFT_REPORT_PATH)

            if summary["production_mae"] is not None:
                mlflow.log_artifact(PERFORMANCE_REPORT_PATH)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    monitor_model()
