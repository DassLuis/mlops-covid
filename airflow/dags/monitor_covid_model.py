import json
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator


PROJECT_DIR = "/opt/airflow/project"
SUMMARY_PATH = f"{PROJECT_DIR}/data/monitoring/monitoring_summary.json"

default_args = {
    "owner": "mlops-covid",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def choose_next_step():
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        summary = json.load(f)

    if summary["retrain_recommended"]:
        return "trigger_retraining"

    return "no_action"


with DAG(
    dag_id="monitor_covid_model",
    description="Suivi de la dérive et de la performance du modèle en production",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "covid", "monitoring"],
) as dag:

    # Rapports Evidently : dérive des entrées et performance réelle du modèle
    run_monitoring = BashOperator(
        task_id="run_monitoring",
        bash_command=f"cd {PROJECT_DIR} && python src/monitor.py",
    )

    check_results = BranchPythonOperator(
        task_id="check_results",
        python_callable=choose_next_step,
    )

    # Dérive ou baisse de performance : on relance la chaîne d'entraînement
    trigger_retraining = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="retrain_covid_model",
    )

    no_action = EmptyOperator(task_id="no_action")

    run_monitoring >> check_results >> [trigger_retraining, no_action]
