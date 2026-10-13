from datetime import datetime, timedelta

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator


TRAINING_URL = "http://training:8000"

default_args = {
    "owner": "mlops-covid",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def training_step(name: str, query: str = "") -> str:
    """Airflow orchestre ; chaque étape s'exécute dans le service d'entraînement."""
    return (
        "curl --fail-with-body --silent --show-error --max-time 3600 -X POST "
        f'-H "x-api-key: $API_KEY" "{TRAINING_URL}/steps/{name}{query}"'
    )


with DAG(
    dag_id="retrain_covid_model",
    description="Réentraînement, validation et promotion du modèle COVID",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule_interval="@weekly",
    catchup=False,
    max_active_runs=1,
    params={
        "force_train": Param(
            False,
            type="boolean",
            description="Réentraîner même si ni les données ni le code n'ont changé",
        ),
    },
    tags=["mlops", "covid", "training"],
) as dag:

    # Les étapes sont celles de dvc.yaml : DVC ne rejoue que ce qui a changé
    ingest_data = BashOperator(
        task_id="ingest_data",
        bash_command=training_step("ingest"),
    )

    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=training_step("prepare"),
    )

    # Enregistre une nouvelle version « candidate » dans le registre MLflow
    train_model = BashOperator(
        task_id="train_model",
        bash_command=training_step(
            "train", "?force={{ 'true' if params.force_train else 'false' }}"
        ),
    )

    # Compare le candidat au modèle en production et ne promeut que le meilleur
    promote_model = BashOperator(
        task_id="promote_model",
        bash_command=training_step("promote"),
    )

    version_artifacts = BashOperator(
        task_id="version_artifacts",
        bash_command=training_step("push"),
    )

    reload_api = BashOperator(
        task_id="reload_api",
        bash_command=(
            'curl --fail --silent --show-error -X POST '
            '-H "x-api-key: $API_KEY" http://api:8000/model/reload'
        ),
    )

    (
        ingest_data
        >> prepare_data
        >> train_model
        >> promote_model
        >> version_artifacts
        >> reload_api
    )
