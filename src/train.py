import os
import sys
import json
import hashlib
import subprocess

import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from dotenv import load_dotenv
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import FEATURES, TARGET
from src.config import (
    COUNTRY,
    START_DATE,
    END_DATE,
    HORIZON,
    TRAIN_RATIO,
    N_ESTIMATORS,
    RANDOM_STATE,
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    METRICS_PATH,
    MLFLOW_EXPERIMENT_NAME,
)

load_dotenv()


def file_md5(path: str) -> str:
    """Empreinte du fichier, identique à celle que DVC inscrit dans dvc.lock."""
    md5 = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5.update(chunk)

    return md5.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "inconnu"


def train_model():
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print("Chargement des données préparées...")
    df_model = pd.read_csv(PROCESSED_DATA_PATH)
    data_md5 = file_md5(PROCESSED_DATA_PATH)

    features = list(FEATURES)

    X = df_model[features]
    y = df_model[TARGET]

    train_size = int(len(df_model) * TRAIN_RATIO)

    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]

    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    print("Entraînement du modèle...")
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    with mlflow.start_run() as run:
        mlflow.log_param("country", COUNTRY)
        mlflow.log_param("start_date", START_DATE)
        mlflow.log_param("end_date", END_DATE)
        mlflow.log_param("horizon", HORIZON)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("train_size", train_size)
        mlflow.log_param("features", features)

        # Lien entre le run et la version des données suivie par DVC
        mlflow.set_tag("dvc_data_md5", data_md5)
        mlflow.set_tag("dvc_data_path", PROCESSED_DATA_PATH)
        mlflow.set_tag("git_commit", git_commit())

        dataset = mlflow.data.from_pandas(
            df_model,
            source=PROCESSED_DATA_PATH,
            name="covid_features",
            targets=TARGET,
            digest=data_md5[:8],
        )
        mlflow.log_input(dataset, context="training")

        model.fit(X_train, y_train)

        print("Évaluation du modèle...")
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"MAE : {mae:,.2f}")
        print(f"RMSE : {rmse:,.2f}")

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

        model_package = {
            "model": model,
            "features": features,
            "country": COUNTRY,
            "horizon": HORIZON,
            "mae": mae,
            "rmse": rmse,
        }

        joblib.dump(model_package, MODEL_PATH)

        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(X_train, y_pred),
            input_example=X_train.head(1),
        )

        os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

        metrics = {
            "mae": float(mae),
            "rmse": float(rmse),
            "train_rows": int(train_size),
            "test_rows": int(len(df_model) - train_size),
            "data_md5": data_md5,
            "mlflow_run_id": run.info.run_id,
        }

        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"Modèle sauvegardé dans {MODEL_PATH}")
        print(f"Run MLflow : {run.info.run_id}")


if __name__ == "__main__":
    train_model()
