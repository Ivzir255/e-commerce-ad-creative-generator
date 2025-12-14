
# airflow/dags/dvc_training_pipeline_dag.py
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta
import os
import json

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email': ['alerts@company.com'],
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

REPO_PATH = '/opt/airflow/repo'

def check_if_retraining_needed(**context):
    """Check if model retraining is needed based on data drift or metrics"""
    # Load current metrics
    metrics_path = f'{REPO_PATH}/metrics.json'
    
    if not os.path.exists(metrics_path):
        print("No existing metrics found. Triggering training.")
        return True
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Check if metrics are below threshold
    eval_loss_threshold = 0.5
    if metrics.get('eval_loss', 1.0) > eval_loss_threshold:
        print(f"Eval loss {metrics['eval_loss']} above threshold. Triggering retraining.")
        return True
    
    # Check data version
    # In production, you'd check data drift, distribution shifts, etc.
    print("Model performance is good. Skipping retraining.")
    return False

with DAG(
    'dvc_training_pipeline',
    default_args=default_args,
    description='DVC-tracked model training pipeline',
    schedule_interval='@weekly',
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['dvc', 'training', 'model'],
) as dag:
    
    # Wait for data ingestion to complete
    wait_for_data = ExternalTaskSensor(
        task_id='wait_for_data_ingestion',
        external_dag_id='dvc_data_ingestion_pipeline',
        external_task_id='log_data_metrics',
        timeout=3600,
        mode='reschedule',
    )
    
    # Pull latest code and data
    git_pull = BashOperator(
        task_id='git_pull',
        bash_command=f'cd {REPO_PATH} && git pull origin main',
    )
    
    # Pull all DVC tracked files (data + models)
    dvc_pull = BashOperator(
        task_id='dvc_pull_all',
        bash_command=f'cd {REPO_PATH} && dvc pull',
    )
    
    # Check if retraining is needed
    check_retraining = PythonOperator(
        task_id='check_retraining_needed',
        python_callable=check_if_retraining_needed,
        provide_context=True,
    )
    
    # Run DVC pipeline (training stage)
    run_training = BashOperator(
        task_id='run_model_training',
        bash_command=f'''
            cd {REPO_PATH}
            # Run training stage
            dvc repro train_model
            
            # Run evaluation stage
            dvc repro evaluate_model
        ''',
    )
    
    # Push trained model and outputs
    dvc_push = BashOperator(
        task_id='dvc_push_model',
        bash_command=f'''
            cd {REPO_PATH}
            dvc push
        ''',
    )
    
    # Commit changes (dvc.lock, metrics, plots)
    git_commit = BashOperator(
        task_id='git_commit_training_results',
        bash_command=f'''
            cd {REPO_PATH}
            git config user.name "Airflow"
            git config user.email "airflow@mlops.com"
            git add dvc.lock metrics.json evaluation_metrics.json plots/
            git add models/final_model.dvc
            git diff --quiet && git diff --staged --quiet || git commit -m "Training run: $(date +%Y-%m-%d) [automated]"
            git push origin main
        ''',
    )
    
    # Tag successful training
    git_tag = BashOperator(
        task_id='tag_model_version',
        bash_command=f'''
            cd {REPO_PATH}
            VERSION="model-$(date +%Y%m%d-%H%M%S)"
            git tag -a $VERSION -m "Automated model training"
            git push origin $VERSION
            echo "Tagged as: $VERSION"
        ''',
    )
    
    # Create DVC experiment
    create_experiment = BashOperator(
        task_id='create_dvc_experiment',
        bash_command=f'''
            cd {REPO_PATH}
            dvc exp save -n "airflow-training-$(date +%Y%m%d-%H%M%S)"
        ''',
    )
    
    # Trigger deployment (optional)
    trigger_deploy = BashOperator(
        task_id='trigger_deployment',
        bash_command=f'''
            echo "Triggering deployment pipeline..."
            # In production: trigger GitHub Actions or update K8s deployment
            curl -X POST http://jenkins:8080/job/deploy-model/build \
                -H "Authorization: Bearer $JENKINS_TOKEN" || true
        ''',
    )
    
    # Define task dependencies
    wait_for_data >> git_pull >> dvc_pull >> check_retraining
    check_retraining >> run_training >> dvc_push
    dvc_push >> git_commit >> git_tag >> create_experiment >> trigger_deploy


# airflow/dags/dvc_experiment_tracking_dag.py
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import yaml

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

REPO_PATH = '/opt/airflow/repo'

# Different hyperparameter configurations to try
EXPERIMENTS = [
    {'training.learning_rate': 1e-5, 'training.batch_size': 4},
    {'training.learning_rate': 5e-5, 'training.batch_size': 8},
    {'training.learning_rate': 1e-4, 'training.batch_size': 16},
]

def create_experiment_task(exp_name, params):
    """Create a DVC experiment with given parameters"""
    param_str = ' '.join([f'--set-param {k}={v}' for k, v in params.items()])
    
    return BashOperator(
        task_id=f'experiment_{exp_name}',
        bash_command=f'''
            cd {REPO_PATH}
            dvc exp run --name {exp_name} {param_str}
        ''',
    )

with DAG(
    'dvc_experiment_tracking',
    default_args=default_args,
    description='Run multiple DVC experiments with different hyperparameters',
    schedule_interval=None,  # Manual trigger
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['dvc', 'experiments', 'hyperparameter-tuning'],
) as dag:
    
    # Setup
    setup = BashOperator(
        task_id='setup',
        bash_command=f'''
            cd {REPO_PATH}
            git pull origin main
            dvc pull
        ''',
    )
    
    # Run experiments in parallel
    experiments = []
    for i, params in enumerate(EXPERIMENTS):
        exp_name = f'exp_{i}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        exp_task = create_experiment_task(exp_name, params)
        experiments.append(exp_task)
        setup >> exp_task
    
    # Compare all experiments
    compare_experiments = BashOperator(
        task_id='compare_experiments',
        bash_command=f'''
            cd {REPO_PATH}
            dvc exp show --csv > experiments_comparison.csv
            cat experiments_comparison.csv
        ''',
    )
    
    # Select and apply best experiment
    apply_best = BashOperator(
        task_id='apply_best_experiment',
        bash_command=f'''
            cd {REPO_PATH}
            # Get best experiment by eval_loss
            BEST_EXP=$(dvc exp show --csv | tail -n +2 | sort -t',' -k5 -n | head -1 | cut -d',' -f1)
            echo "Best experiment: $BEST_EXP"
            dvc exp apply $BEST_EXP
            
            # Commit best experiment
            git add params.yaml dvc.lock metrics.json
            git commit -m "Apply best experiment: $BEST_EXP"
            git push origin main
        ''',
    )
    
    # Set dependencies
    for exp in experiments:
        exp >> compare_experiments
    
    compare_experiments >> apply_best