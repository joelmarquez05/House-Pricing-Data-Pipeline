from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os
import yaml


BASE_DIR = os.getcwd()
with open(os.path.join(BASE_DIR, "config.yaml"), 'r') as f:
    config = yaml.safe_load(f)

VENV_PYTHON = os.path.join(BASE_DIR, config['paths']['venv_subpath'])
SCRIPTS_DIR = os.path.join(BASE_DIR, config['paths']['scripts_dir'])

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email': ['user@example.com'],
    'email_on_failure': True,   # Alerts on failure
    'email_on_retry': False,
    'retries': 3,               # Automatic retries
    'retry_delay': timedelta(minutes=5),
}

# DAG Definition
with DAG(
    dag_id='bcn_housing_pipeline',
    default_args=default_args,
    description='End-to-end BCN Housing Data Pipeline',
    schedule='@daily',
    catchup=False
) as dag:

    # Task 1: Ingestion (Landing Zone)
    ingestion_task = BashOperator(
        task_id='ingestion',
        bash_command=f"{VENV_PYTHON} {os.path.join(SCRIPTS_DIR, 'a3_ingestion.py')}",
        cwd=BASE_DIR,
    )

    # Task 2: Formatting (Formatted Zone)
    formatting_task = BashOperator(
        task_id='formatting',
        bash_command=f"{VENV_PYTHON} {os.path.join(SCRIPTS_DIR, 'a4_formatting.py')}",
        cwd=BASE_DIR,
    )

    # Task 3: Exploitation (Exploitation Zone)
    exploitation_task = BashOperator(
        task_id='exploitation',
        bash_command=f"{VENV_PYTHON} {os.path.join(SCRIPTS_DIR, 'a5_explotation.py')}",
        cwd=BASE_DIR,
    )

    # Task 4: Training (ML Mode)
    training_task = BashOperator(
        task_id='training',
        bash_command=f"{VENV_PYTHON} {os.path.join(SCRIPTS_DIR, 'b1_training.py')}",
        cwd=BASE_DIR,
    )

    # Task 5: MLflow (Model Management)
    mlflow_task = BashOperator(
        task_id='mlflow',
        bash_command=f"{VENV_PYTHON} {os.path.join(SCRIPTS_DIR, 'b2_3_mlflow.py')}",
        cwd=BASE_DIR,
    )

    # Dependency Management
    ingestion_task >> formatting_task >> exploitation_task >> training_task >> mlflow_task