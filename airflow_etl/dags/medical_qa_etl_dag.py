"""
medical_qa_etl_dag.py
A minimal Airflow DAG to demonstrate an ETL + EDA pipeline for medical Q&A data.
"""

import os
from datetime import datetime, timedelta, UTC

from airflow import DAG
from airflow.operators.python import PythonOperator

# 1. Define your Python callables for each step
# Note: All datetime objects in context will be timezone-aware (UTC)
def extract_medquad_callable(**context):
    import subprocess
    subprocess.run(["python", "scripts/etl_medquad.py"], check=True)

def extract_icliniq_callable(**context):
    import subprocess
    subprocess.run(["python", "scripts/etl_icliniq.py"], check=True)

def merge_and_clean_callable(**context):
    import subprocess
    subprocess.run(["python", "scripts/merge_and_clean.py"], check=True)

def eda_analysis_callable(**context):
    import subprocess
    subprocess.run(["python", "scripts/EDA.py"], check=True)

# 2. Default DAG args
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
    # Ensure timezone-aware execution dates
    'execution_timeout': timedelta(hours=1),
}

# 3. Instantiate the DAG
with DAG(
    'medical_qa_etl_dag',
    default_args=default_args,
    description='ETL + EDA pipeline for Medical QA data',
    schedule=None,   # or '0 0 * * *' for daily
    start_date=datetime(2023, 1, 1, tzinfo=UTC),
    catchup=False
) as dag:

    extract_medquad = PythonOperator(
        task_id='extract_medquad',
        python_callable=extract_medquad_callable,
    )

    extract_icliniq = PythonOperator(
        task_id='extract_icliniq',
        python_callable=extract_icliniq_callable,
    )

    merge_and_clean = PythonOperator(
        task_id='merge_and_clean',
        python_callable=merge_and_clean_callable,
    )

    eda_analysis = PythonOperator(
        task_id='eda_analysis',
        python_callable=eda_analysis_callable,
    )

    # 4. Set dependencies
    [extract_medquad, extract_icliniq] >> merge_and_clean >> eda_analysis
    print('dag run complete')
