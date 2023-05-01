import sys
sys.path.insert(0, '/home/kaliber/airflow')

from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import task

default_args = {
    'owner': 'kaliber',
    'start_date': datetime(2023, 4, 29),
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id="forecast_etl_dag",
    default_args=default_args,
    description="DAG",
    start_date=datetime(2023, 4, 29),
    schedule_interval="@daily"
) as dag:
    from projs.exam_etl.forecast_etl import ForecastETL

    @task()
    def run_entire_program():
        api_call = ForecastETL()
        api_call.transform_forecast()
        api_call.save_files()
        api_call.init_db()

    run_entire_program = run_entire_program()
