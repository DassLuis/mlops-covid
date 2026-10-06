import json
import os
import time

from fastapi import Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from src.config import MONITORING_SUMMARY_PATH


REQUESTS = Counter(
    "covid_api_requests_total",
    "Nombre de requêtes reçues par l'API",
    ["method", "endpoint", "status"],
)

REQUEST_DURATION = Histogram(
    "covid_api_request_duration_seconds",
    "Durée de traitement des requêtes",
    ["endpoint"],
)

PREDICTIONS = Counter(
    "covid_predictions_total",
    "Nombre de prédictions servies",
    ["model_version"],
)

PREDICTION_VALUE = Histogram(
    "covid_prediction_value",
    "Distribution des prédictions (nouveaux cas à J+7)",
    buckets=(0, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000),
)

MODEL_INFO = Gauge(
    "covid_model_info",
    "Modèle actuellement chargé par l'API (valeur toujours égale à 1)",
    ["version", "source"],
)

MODEL_LOADED = Gauge("covid_model_loaded", "1 si un modèle est chargé, 0 sinon")
MODEL_TRAINING_MAE = Gauge("covid_model_training_mae", "MAE du modèle chargé, mesurée à l'entraînement")

# Résultats du dernier contrôle Evidently (src/monitor.py), relus à chaque collecte
DRIFT_SHARE = Gauge("covid_data_drift_share", "Part des variables en dérive")
DRIFTED_FEATURES = Gauge("covid_drifted_features", "Nombre de variables en dérive")
DRIFT_DETECTED = Gauge("covid_data_drift_detected", "1 si une dérive des données est détectée")
PRODUCTION_MAE = Gauge("covid_production_mae", "MAE mesurée sur les prédictions servies en production")
PERFORMANCE_DEGRADED = Gauge("covid_performance_degraded", "1 si la performance en production est dégradée")
MONITORING_LAST_RUN = Gauge("covid_monitoring_last_run_timestamp", "Horodatage du dernier contrôle de monitoring")


def set_model_metrics(model_package):
    MODEL_INFO.clear()

    if model_package is None:
        MODEL_LOADED.set(0)
        return

    MODEL_LOADED.set(1)
    MODEL_INFO.labels(
        version=model_package["version"],
        source=model_package["source"],
    ).set(1)
    MODEL_TRAINING_MAE.set(float(model_package["mae"]))


def record_prediction(prediction: float, model_version: str):
    PREDICTIONS.labels(model_version=model_version).inc()
    PREDICTION_VALUE.observe(prediction)


def refresh_monitoring_metrics():
    path = os.getenv("MONITORING_SUMMARY_PATH", MONITORING_SUMMARY_PATH)

    try:
        with open(path, encoding="utf-8") as f:
            summary = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return

    gauges = {
        DRIFT_SHARE: summary.get("drift_share"),
        DRIFTED_FEATURES: summary.get("drifted_features_count"),
        DRIFT_DETECTED: summary.get("drift_detected"),
        PRODUCTION_MAE: summary.get("production_mae"),
        PERFORMANCE_DEGRADED: summary.get("performance_degraded"),
        MONITORING_LAST_RUN: summary.get("timestamp_unix"),
    }

    for gauge, value in gauges.items():
        if value is not None:
            gauge.set(float(value))


async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    status = 500

    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        # Le gabarit de route (et non l'URL brute) évite de multiplier les séries
        route = request.scope.get("route")
        endpoint = route.path if route else "inconnu"

        REQUESTS.labels(
            method=request.method,
            endpoint=endpoint,
            status=str(status),
        ).inc()
        REQUEST_DURATION.labels(endpoint=endpoint).observe(time.perf_counter() - start)


def metrics_endpoint():
    refresh_monitoring_metrics()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
