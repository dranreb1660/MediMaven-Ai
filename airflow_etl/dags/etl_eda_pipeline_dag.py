"""
medical_qa_etl_dag.py
A minimal Airflow DAG to demonstrate an ETL + EDA pipeline for medical Q&A data.
"""

import os
from datetime import datetime, timedelta, UTC

from airflow import DAG
from airflow.operators.python import PythonOperator


# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,  # This ensures each run executes independently
    'start_date': datetime(2025, 3, 1),  # Adjust to match deployment date
    'retries': 1,  # Number of retries if the task fails
    'retry_delay': timedelta(minutes=5),  # Time to wait before retrying
}

def run_etl_pipeline():
    """Triggers the ZenML ETL pipeline to process and store data."""
    from pipelines.etl_pipeline import etl_pipeline
    etl_pipeline().run()


def run_eda_pipeline():
    """Triggers the ZenML EDA pipeline after ETL is complete."""
    from pipelines.eda_pipeline import eda_pipeline
    eda_pipeline().run()

# 3. Instantiate the DAG

with DAG(
    dag_id='etl_eda_pipeline_dag',  # Single DAG for ETL + EDA
    default_args=default_args,
    schedule='@daily',  # Runs once per day (modify as needed)
    catchup=False,  # Ensures past runs are not executed retroactively
) as dag:
    
    
    # Task 1: Run ETL pipeline
    run_etl = PythonOperator(
        task_id='run_etl',
        python_callable=run_etl_pipeline,
    )

    # Task 2: Run EDA pipeline (Only after ETL completes successfully)
    run_eda = PythonOperator(
        task_id='run_eda',
        python_callable=run_eda_pipeline,
    )


    # Define task dependencies
    run_etl >> run_eda  # EDA runs only after ETL completes successfully
