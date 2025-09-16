from airflow                                   import DAG
from airflow.providers.standard.operators.bash import BashOperator
from datetime                                  import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'financial_data_ingestion',
    default_args=default_args,
    description='Ingest FMP API and MySQL to S3',
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False
)

ingest_fmp = BashOperator(
    task_id='ingest_fmp',
    bash_command='/home/codezeus/project_airflow/aws-financial-pipeline/scripts/ingest_fmp.py',
    dag=dag
)

ingest_mysql = BashOperator(
    task_id='ingest_mysql',
    bash_command='/home/codezeus/project_airflow/aws-financial-pipeline/scripts/ingest_mysql.py',
    dag=dag
)

# ingest_fmp >> ingest_mysql