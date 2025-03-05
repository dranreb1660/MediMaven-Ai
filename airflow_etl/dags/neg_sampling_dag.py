from datetime import datetime, timedelta, UTC

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,  # This ensures each run executes independently
    'start_date': datetime(2025, 3, 1),  # Adjust to match deployment date
    'retries': 1,  # Number of retries if the task fails
    'retry_delay': timedelta(minutes=5),  # Time to wait before retrying

}

def run_neg_sampling_pipeline():
    """Triggers the ZenML ETL pipeline to process and store data."""
    from pipelines.neg_sampling_pipeline import neg_sampling_pipeline
    neg_sampling_pipeline().run()


with DAG(
    dag_id='neg_sampling_pipeline_dag',  # Single DAG for ETL + EDA
    default_args=default_args,
    schedule='@daily',  # Runs once per day (modify as needed)
    catchup=False,  # Ensures past runs are not executed retroactively
) as dag:
    
    # Task 1: Run Neg_sampling pipeline
    run_neg_sampling = PythonOperator(
        task_id='run_neg_sampling',
        python_callable=run_neg_sampling_pipeline,
    )