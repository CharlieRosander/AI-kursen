from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "retries": 5,
    "retry_delay": timedelta(minutes=2)
}

with DAG(
    dag_id="our_first_dag_v5",
    default_args=default_args,
    description="Our first DAG",
    start_date=datetime(2023, 4, 25),
    schedule_interval="@daily"
) as dag:
    task1 = BashOperator(
        task_id="first_task",
        bash_command="echo 'Hello World'"
    )

    task2 = BashOperator(
        task_id="second_task",
        bash_command="echo 'I am the second task and I depend on the first task'",
    )

    task3 = BashOperator(
        task_id="third_task",
        bash_command="echo 'I am the third task and I depend on the first task'",
    )

    # Task dependency method 1
    # task1.set_downstream(task2)
    # task1.set_downstream(task3)

    # Task dependency method 2
    # task1 >> task2
    # task1 >> task3

    # Task dependency method 3
    # task1 >> [task2, task3]
    task1 >> task2 >> task3