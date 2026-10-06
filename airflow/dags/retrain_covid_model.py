from datetime import datetime, timedelta

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator


PROJECT_DIR = "cd /opt/airflow/project"

default_args = {
    "owner": "mlops-covid",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


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
        bash_command=f"{PROJECT_DIR} && dvc repro --single-item --force ingest",
    )

    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=f"{PROJECT_DIR} && dvc repro --single-item prepare",
    )

    # Enregistre une nouvelle version « candidate » dans le registre MLflow
    train_model = BashOperator(
        task_id="train_model",
        bash_command=(
            f"{PROJECT_DIR} && dvc repro --single-item "
            "{{ '--force' if params.force_train else '' }} train"
        ),
    )

    # Compare le candidat au modèle en production et ne promeut que le meilleur
    promote_model = BashOperator(
        task_id="promote_model",
        bash_command=f"{PROJECT_DIR} && python src/promote.py",
    )

    version_artifacts = BashOperator(
        task_id="version_artifacts",
        bash_command=f"{PROJECT_DIR} && dvc push",
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
