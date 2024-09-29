import os
import io
import sys
import logging
import numpy as np
import pandas as pd
import pickle

from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score
from typing import NoReturn

from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from dags_config import *

sys.path.append(f"{PROJECT_ROOT}/src")
from data_extract import extract_data
from data_prepare import (
    prepare_data_for_transformer,
    prepare_data_for_tf_idf,
    make_transformer_embeddins,
)

from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score

from train_models import (
    make_train_operator_for_tf_idf,
    make_train_operator_for_transformer,
)


DEFAULT_ARGS = {
    "retry": 3,
    "retry_delay": timedelta(minutes=1),
}

dag = DAG(
    dag_id="train_pipeline",
    schedule="0 1 * * *",
    start_date=days_ago(2),
    catchup=False,
    tags=["pmldl assignment 1"],
    default_args=DEFAULT_ARGS,
)


def init() -> NoReturn:
    LOGGER.info(f"Initialisation:")
    LOGGER.info(f"Working directory: {PROJECT_ROOT}")
    LOGGER.info(f"Airflow home: {os.environ['AIRFLOW_HOME']}")


task_init = PythonOperator(task_id="init", python_callable=init, dag=dag)

task_extract_data = PythonOperator(
    # extract_data
    task_id="extract_data",
    python_callable=extract_data,
    dag=dag,
)

task_prepare_data_for_tf_idf = PythonOperator(
    # prepare_data_for_tf_idf
    task_id="prepare_data_for_tf_idf",
    python_callable=prepare_data_for_tf_idf,
    dag=dag,
)

task_prepare_data_for_transformer = PythonOperator(
    # prepare_data_for_transformer
    task_id="prepare_data_for_transformer",
    python_callable=prepare_data_for_transformer,
    dag=dag,
)

task_make_transformer_embeddins = PythonOperator(
    # make_transformer_embeddins
    task_id="make_transformer_embeddins",
    python_callable=relax,
    dag=dag,
)

tf_idf_tasks = [
    make_train_operator_for_tf_idf(model_params, CONFIG["metrics"], dag)
    for model_params in CONFIG["models"]["tf_idf"]
]

transformer_tasks = [
    make_train_operator_for_transformer(model_params, CONFIG["metrics"], dag)
    for model_params in CONFIG["models"]["transformer"]
]

(
    task_init
    >> task_extract_data
    >> [task_prepare_data_for_transformer, task_prepare_data_for_tf_idf]
)

task_prepare_data_for_transformer.set_downstream(task_make_transformer_embeddins)

# temp = task_prepare_data_for_tf_idf
for task in tf_idf_tasks:
    # temp.set_downstream(task)
    # temp = task
    task_prepare_data_for_tf_idf.set_downstream(task)

# temp = task_make_transformer_embeddins
for task in transformer_tasks:  
    # temp.set_downstream(task)
    # temp = task
    task_make_transformer_embeddins.set_downstream(task)
