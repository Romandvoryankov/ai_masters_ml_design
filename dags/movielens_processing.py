from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime, timedelta
import requests
import zipfile
import os
import shutil
from minio import Minio

# Configuration
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DOWNLOAD_DIR = "/tmp/movielens"
BUCKET_NAME = "movielens"
MINIO_ENDPOINT = "minio:9000"
MINIO_ACCESS_KEY = "minioaccesskey"
MINIO_SECRET_KEY = "miniosecretkey"

# Default arguments for DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def download_and_extract():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    zip_path = os.path.join(DOWNLOAD_DIR, "ml-latest-small.zip")

    # Download dataset
    with requests.get(MOVIELENS_URL, stream=True) as r:
        with open(zip_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    # Extract dataset
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DOWNLOAD_DIR)

    os.remove(zip_path)


def split_datasets():
    ratings_path = os.path.join(DOWNLOAD_DIR, "ml-latest-small", "ratings.csv")
    train_path = os.path.join(DOWNLOAD_DIR, "train.csv")
    test_path = os.path.join(DOWNLOAD_DIR, "test.csv")

    import pandas as pd
    ratings = pd.read_csv(ratings_path)

    # Split into train and test sets by timestamp
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    train = ratings[ratings['timestamp'] < '2018-01-01']
    test = ratings[ratings['timestamp'] >= '2018-01-01']

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)


def upload_to_minio():
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )

    if not client.bucket_exists(BUCKET_NAME):
        client.make_bucket(BUCKET_NAME)

    for root, _, files in os.walk(DOWNLOAD_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            client.fput_object(BUCKET_NAME, file, file_path)

# DAG Definition
dag = DAG(
    'movielens_processing',
    default_args=default_args,
    description='Process Movielens data',
    schedule_interval=None,
    start_date=datetime(2023, 12, 1),
    catchup=False,
)

download_task = PythonOperator(
    task_id='download_and_extract',
    python_callable=download_and_extract,
    dag=dag,
)

split_task = PythonOperator(
    task_id='split_datasets',
    python_callable=split_datasets,
    dag=dag,
)

upload_task = PythonOperator(
    task_id='upload_to_minio',
    python_callable=upload_to_minio,
    dag=dag,
)

spark_train_task = SparkSubmitOperator(
    task_id='train_model',
    application="/opt/spark/jobs/train_model.py",
    conn_id="spark_default",
    conf={
        "spark.master": "spark://spark-master:7077",  # или local[*]
        "spark.executor.memory": "2g"
    },
    dag=dag,
)

spark_test_task = SparkSubmitOperator(
    task_id='test_model',
    application="/opt/spark/jobs/test_model.py",
    conn_id="spark_default",
    conf={
        "spark.master": "spark://spark-master:7077",  # или local[*]
        "spark.executor.memory": "2g"
    },
    dag=dag,
)

# Task dependencies
download_task >> split_task >> upload_task >> spark_train_task >> spark_test_task
