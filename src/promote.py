import os
import sys
import json

import pandas as pd
import mlflow
import mlflow.sklearn

from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import mean_absolute_error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import FEATURES, TARGET
from src.config import (
    TRAIN_RATIO,
    PROCESSED_DATA_PATH,
    PROMOTION_PATH,
    MLFLOW_MODEL_NAME,
    ALIAS_CANDIDATE,
    ALIAS_PRODUCTION,
)

load_dotenv()


def get_version(client: MlflowClient, alias: str):
    try:
        return client.get_model_version_by_alias(MLFLOW_MODEL_NAME, alias)
    except MlflowException:
        return None


def evaluate_version(version, X_test, y_test) -> float:
    model = mlflow.sklearn.load_model(f"models:/{MLFLOW_MODEL_NAME}/{version.version}")
    return float(mean_absolute_error(y_test, model.predict(X_test)))


def promote_model():
    """Compare le modèle candidat au modèle en production sur le même jeu de test,
    et ne le promeut que s'il fait au moins aussi bien."""
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    client = MlflowClient()

    candidate = get_version(client, ALIAS_CANDIDATE)

    if candidate is None:
        raise RuntimeError(
            f"Aucune version '{ALIAS_CANDIDATE}' dans le registre : lancer src/train.py d'abord."
        )

    production = get_version(client, ALIAS_PRODUCTION)

    df_model = pd.read_csv(PROCESSED_DATA_PATH)
    train_size = int(len(df_model) * TRAIN_RATIO)

    X_test = df_model[list(FEATURES)].iloc[train_size:]
    y_test = df_model[TARGET].iloc[train_size:]

    candidate_mae = evaluate_version(candidate, X_test, y_test)
    production_mae = None

    if production is None:
        promoted = True
        reason = "aucun modèle en production"
    elif production.version == candidate.version:
        promoted = False
        production_mae = candidate_mae
        reason = "le candidat est déjà en production"
    else:
        production_mae = evaluate_version(production, X_test, y_test)
        promoted = candidate_mae <= production_mae
        reason = (
            "MAE du candidat inférieure ou égale à celle de la production"
            if promoted
            else "MAE du candidat supérieure à celle de la production"
        )

    if promoted:
        client.set_registered_model_alias(
            MLFLOW_MODEL_NAME, ALIAS_PRODUCTION, candidate.version
        )
        client.set_model_version_tag(
            MLFLOW_MODEL_NAME, candidate.version, "validation_status", "promu"
        )

        if production is not None:
            client.set_model_version_tag(
                MLFLOW_MODEL_NAME, production.version, "validation_status", "remplacé"
            )
    elif production.version != candidate.version:
        client.set_model_version_tag(
            MLFLOW_MODEL_NAME, candidate.version, "validation_status", "rejeté"
        )

    result = {
        "candidate_version": str(candidate.version),
        "candidate_mae": candidate_mae,
        "previous_production_version": str(production.version) if production else None,
        "previous_production_mae": production_mae,
        "promoted": promoted,
        "reason": reason,
        "production_version": str(candidate.version if promoted else production.version),
    }

    os.makedirs(os.path.dirname(PROMOTION_PATH), exist_ok=True)

    with open(PROMOTION_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    promote_model()
