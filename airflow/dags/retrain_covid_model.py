from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


default_args = {
    "owner": "mlops-covid",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="retrain_covid_model",
    description="Réentraînement automatique du modèle COVID",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule_interval="@weekly",
    catchup=False,
    tags=["mlops", "covid", "training"],
) as dag:

    retrain_model = BashOperator(
        task_id="retrain_random_forest_model",
        bash_command="cd /opt/airflow/project && python src/train.py",
    )

    retrain_model