import json

import joblib
import pandas as pd

from src.config import MODEL_PATH, MLFLOW_MODEL_NAME, ALIAS_PRODUCTION


def load_model(model_path: str = MODEL_PATH):
    model_package = joblib.load(model_path)
    model_package.setdefault("version", "local")
    model_package.setdefault("source", "fichier local")
    return model_package


def load_model_from_registry(
    model_name: str = MLFLOW_MODEL_NAME,
    alias: str = ALIAS_PRODUCTION,
):
    """Charge la version du registre MLflow qui porte l'alias demandé."""
    import mlflow.sklearn
    from mlflow import MlflowClient

    version = MlflowClient().get_model_version_by_alias(model_name, alias)
    model = mlflow.sklearn.load_model(f"models:/{model_name}@{alias}")

    return {
        "model": model,
        "features": json.loads(version.tags["features"]),
        "country": version.tags["country"],
        "horizon": int(version.tags["horizon"]),
        "mae": float(version.tags["mae"]),
        "rmse": float(version.tags["rmse"]),
        "version": str(version.version),
        "source": f"registre MLflow ({model_name}@{alias})",
    }


def make_prediction(input_data: dict, model_package: dict):
    model = model_package["model"]
    features = model_package["features"]

    input_df = pd.DataFrame([input_data])
    input_df = input_df[features]

    prediction = model.predict(input_df)[0]

    return float(prediction)
