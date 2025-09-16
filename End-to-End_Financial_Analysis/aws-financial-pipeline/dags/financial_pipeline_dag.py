# File: dags/financial_pipeline_dag.py
# Purpose: Orchestrate full pipeline - from ingestion to production
# It is adviseable to use the @task over the PythonOperator
# https://airflow.apache.org/docs/apache-airflow/2.11.0/howto/operator/python.html

from airflow                                     import DAG
from airflow.operators                           import python
from airflow.providers.amazon.aws.operators.glue import GlueJobOperator
from airflow.utils.dates                         import days_ago
from datetime                                    import timedelta
import os
from runpy                                       import run_path

# Define default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 1,
    'retry_delay': timedelta(minutes=2)
}

dag = DAG(
    'financial_pipeline_dag',
    default_args=default_args,
    description='ETL pipeline for financial data lake',
    schedule_interval=None, #@daily, @weekly, @monthly
    catchup=False
)

# 1. Python Ingestion Scripts
def ingest_fmp():
    run_path("/home/codezeus/project_airflow/aws-financial-pipeline/scripts/ingest_fmp.py")
#

def ingest_mysql():
    run_path("/home/codezeus/project_airflow/aws-financial-pipeline/scripts/ingest_mysql.py")
#

def wrangle_data():
    run_path("/home/codezeus/project_airflow/aws-financial-pipeline/scripts/data_wrangler.py")
#

api_ingestion_task = python.PythonOperator(
    task_id='api_ingestion',
    python_callable=ingest_fmp,
    dag=dag
)

mysql_ingestion_task = python.PythonOperator(
    task_id='mysql_ingestion',
    python_callable=ingest_mysql,
    dag=dag
)

# 2. Trigger Glue ETL (transform)
transform_task = GlueJobOperator(
    task_id='transform_financial_data',
    job_name='transform_financial_data',
    script_location='s3://your-financial-data-lake/scripts/transform_financial_data.py',
    region_name='us-east-2',
    iam_role_name='glue-role',
    dag=dag
)

# 3. Trigger Glue Join Job
join_task =  GlueJobOperator(
    task_id='join_financial_summary',
    job_name='join_financial_summary',
    script_location='s3://your-financial-data-lake/scripts/join_financial_summary.py',
    region_name='us-east-2',
    iam_role_name='glue-role',
    dag=dag
)

# 4. Data wrangling with Python
wrangle_task = python.PythonOperator(
    task_id='data_wrangling',
    python_callable=wrangle_data,
    dag=dag
)
# wrangle_task = GlueJobOperator(
#     task_id='wrangle_data',
#     job_name='data_wrangler',
#     script_location='s3://your-financial-data-lake/scripts/data_wrangler.py',
#     region_name='us-east-2',
#     iam_role_name='glue-role',
#     dag=dag
# )

# Task dependencies
[api_ingestion_task, mysql_ingestion_task] >> transform_task >> join_task >> wrangle_task
