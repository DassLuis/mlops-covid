import os
import sys

from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import load_model, make_prediction
from api.auth import verify_api_key


model_package = load_model()


app = FastAPI(
    title="API COVID - MLOps",
    description="API d'inférence pour prédire les nouveaux cas COVID à H+7.",
    version="2.0.0",
)


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


@app.get("/")
def home():
    return {
        "message": "API COVID opérationnelle",
        "docs": "/docs",
        "predict_endpoint": "/predict",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "country": model_package["country"],
        "horizon": model_package["horizon"],
        "mae": round(float(model_package["mae"]), 2),
        "rmse": round(float(model_package["rmse"]), 2),
        "features": model_package["features"],
    }


@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict(data: CovidInput):
    input_data = data.model_dump()

    prediction = make_prediction(
        input_data=input_data,
        model_package=model_package,
    )

    return {
        "prediction_new_cases_h_plus_7": round(prediction, 2),
    }