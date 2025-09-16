from airflow                import DAG
from airflow.operators.bash import BashOperator
from datetime               import datetime

# Define default args
default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 1, 1),
    "retries": 0,
}

# Define the DAG
with DAG(
    dag_id="simple_test_dag",
    default_args=default_args,
    schedule_interval=None,  # Run manually
    catchup=False,
    tags=["example"],
) as dag:

    # Task 1: Start
    start_task = BashOperator(
        task_id="start",
        bash_command="echo 'Starting the DAG...'"
    )

    # Task 2: Process
    process_task = BashOperator(
        task_id="process",
        bash_command="echo 'Processing some data...'"
    )

    # Task 3: End
    end_task = BashOperator(
        task_id="end",
        bash_command="echo 'DAG completed!'"
    )

    # Define task order
    start_task >> process_task >> end_task
