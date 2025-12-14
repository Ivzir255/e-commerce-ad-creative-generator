# airflow/dags/simple_data_ingestion_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os
import sys

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email': ['alerts@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

REPO_PATH = '/opt/airflow/repo'

with DAG(
    'simple_data_ingestion_pipeline',
    default_args=default_args,
    description='Simple data ingestion pipeline',
    schedule_interval='@daily',
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['data', 'ingestion'],
) as dag:
    
    # Pull latest code
    git_pull = BashOperator(
        task_id='git_pull',
        bash_command=f'cd {REPO_PATH} && git pull origin main',
    )
    
    # Pull existing DVC tracked data (if any)
    dvc_pull = BashOperator(
        task_id='dvc_pull_data',
        bash_command=f'cd {REPO_PATH} && dvc pull || true',  # Don't fail if no DVC data
    )
    
    # Run data ingestion script directly
    run_data_ingestion = BashOperator(
        task_id='run_data_ingestion',
        bash_command=f'''
            cd {REPO_PATH}
            python src/data/data_ingestion.py
        ''',
    )
    
    # Add processed data to DVC tracking
    dvc_add_data = BashOperator(
        task_id='dvc_track_outputs',
        bash_command=f'''
            cd {REPO_PATH}
            dvc add data/raw/products.csv
            dvc add data/processed/products.parquet
            dvc add data/processed/training_data.json
        ''',
    )
    
    # Push data to DVC remote
    dvc_push = BashOperator(
        task_id='dvc_push_results',
        bash_command=f'''
            cd {REPO_PATH}
            dvc push
        ''',
    )
    
    # Commit DVC metadata changes to Git
    git_commit = BashOperator(
        task_id='git_commit_dvc_changes',
        bash_command=f'''
            cd {REPO_PATH}
            git config user.name "Airflow"
            git config user.email "airflow@mlops.com"
            git add data/raw/products.csv.dvc \
                    data/processed/products.parquet.dvc \
                    data/processed/training_data.json.dvc \
                    data/processed/data_stats.json \
                    data/.gitignore
            git diff --quiet && git diff --staged --quiet || \
                git commit -m "Data update: $(date +%Y-%m-%d) [automated]"
            git push origin main || true
        ''',
    )
    
    # Log completion
    log_metrics = PythonOperator(
        task_id='log_completion',
        python_callable=lambda: print("✅ Data pipeline completed successfully"),
    )
    
    # Define task dependencies (simplified linear flow)
    git_pull >> dvc_pull >> run_data_ingestion >> dvc_add_data >> dvc_push >> git_commit >> log_metrics