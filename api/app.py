import os
import sys
from datetime import date as Date
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import load_model, load_model_from_registry, make_prediction
from api.auth import verify_api_key
from api.metrics import (
    metrics_endpoint,
    metrics_middleware,
    record_prediction,
    set_model_metrics,
)
from api.prediction_log import log_prediction


# "registry" : modèle en production du registre MLflow ; "local" : fichier joblib
MODEL_SOURCE = os.getenv("MODEL_SOURCE", "local")


def load_current_model():
    if MODEL_SOURCE == "registry":
        try:
            return load_model_from_registry()
        except Exception as exc:
            print(f"Registre MLflow indisponible ({exc}) : repli sur le fichier local.")

    try:
        return load_model()
    except FileNotFoundError:
        return None


state = {"model_package": load_current_model()}
set_model_metrics(state["model_package"])


app = FastAPI(
    title="API COVID - MLOps",
    description="API d'inférence pour prédire les nouveaux cas COVID à H+7.",
    version="3.0.0",
)

app.middleware("http")(metrics_middleware)
app.get("/metrics", include_in_schema=False)(metrics_endpoint)


class CovidInput(BaseModel):
    lag_1: float = Field(..., example=1200)
    lag_2: float = Field(..., example=1100)
    lag_3: float = Field(..., example=1300)
    lag_4: float = Field(..., example=1250)
    lag_5: float = Field(..., example=1400)
    lag_6: float = Field(..., example=1350)
    lag_7: float = Field(..., example=1500)
    rolling_mean_7: float = Field(..., example=1300)
    new_deaths: float = Field(..., example=25)
    people_vaccinated: float = Field(..., example=40000000)
    stringency_index: float = Field(..., example=45.5)
    # Jour auquel correspondent les données : facultatif, il permet de comparer
    # plus tard la prédiction à la valeur réellement observée 7 jours après.
    date: Optional[Date] = Field(None, example="2023-03-01")


def model_info(model_package: dict):
    return {
        "model_version": model_package["version"],
        "model_source": model_package["source"],
        "country": model_package["country"],
        "horizon": model_package["horizon"],
        "mae": round(float(model_package["mae"]), 2),
        "rmse": round(float(model_package["rmse"]), 2),
        "features": model_package["features"],
    }


@app.get("/")
def home():
    return {
        "message": "API COVID opérationnelle",
        "docs": "/docs",
        "predict_endpoint": "/predict",
    }


@app.get("/health")
def health():
    model_package = state["model_package"]

    if model_package is None:
        return {"status": "degraded", "model_loaded": False}

    return {"status": "ok", "model_loaded": True, **model_info(model_package)}


@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict(data: CovidInput):
    model_package = state["model_package"]

    if model_package is None:
        raise HTTPException(status_code=503, detail="Aucun modèle chargé")

    input_data = data.model_dump(mode="json")

    prediction = make_prediction(
        input_data=input_data,
        model_package=model_package,
    )

    record_prediction(prediction, model_package["version"])
    log_prediction(input_data, prediction, model_package["version"])

    return {
        "prediction_new_cases_h_plus_7": round(prediction, 2),
        "model_version": model_package["version"],
    }


@app.post("/model/reload", dependencies=[Depends(verify_api_key)])
def reload_model():
    """Recharge le modèle en production, après une promotion dans le registre."""
    model_package = load_current_model()

    if model_package is None:
        raise HTTPException(status_code=503, detail="Aucun modèle disponible")

    state["model_package"] = model_package
    set_model_metrics(model_package)

    return {"status": "reloaded", **model_info(model_package)}
