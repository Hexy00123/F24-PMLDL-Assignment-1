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
from data_prepare import prepare_data_for_transformer, prepare_data_for_tf_idf

from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score

from train_models import make_train_operator_for_tf_idf


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

def relax():
    pass 


def init() -> NoReturn:
    LOGGER.info(f"Initialisation:")
    LOGGER.info(f"Working directory: {PROJECT_ROOT}")
    LOGGER.info(f"Airflow home: {os.environ['AIRFLOW_HOME']}")


def train_tf_idf_xgboost() -> NoReturn:
    LOGGER.info(f"Open S3 connection...")
    s3_hook = S3Hook("s3_connector")

    LOGGER.info(f"Downloading preprocessed data from S3...")
    data = {}
    for name in ["X_train", "X_test", "y_train", "y_test"]:
        file = s3_hook.download_file(
            key=DATA_PATH + f"preprocessed/tf_idf/{name}.pkl", bucket_name=BUCKET
        )
        data[name] = np.array(pd.read_pickle(file)).reshape(-1)

    X_train, X_test, y_train, y_test = (
        data["X_train"],
        data["X_test"],
        data["y_train"],
        data["y_test"],
    )

    LOGGER.info(f"Downloaded succesfully...")

    LOGGER.info(f"Training model...")
    pipeline = Pipeline(
        steps=[
            ("tf_idf", TfidfVectorizer(max_features=250)),
            ("classifier", XGBClassifier(n_estimators=150, max_depth=3, device="cpu")),
        ]
    ).fit(X_train, y_train)

    LOGGER.info(f"Evaluating...")
    f1_macro = f1_score(y_test, pipeline.predict(X_test), average="macro")
    f1_weighted = f1_score(y_test, pipeline.predict(X_test), average="weighted")

    LOGGER.info(f"F1 macro: {f1_macro}")
    LOGGER.info(f"F1 weighted: {f1_weighted}")


task_init = PythonOperator(task_id="init", python_callable=init, dag=dag)

task_extract_data = PythonOperator(
    #extract_data
    task_id="extract_data", python_callable=extract_data, dag=dag
)

task_prepare_data_for_transformer = PythonOperator(
    #prepare_data_for_transformer
    task_id="prepare_data_for_transformer",
    python_callable=prepare_data_for_transformer,
    dag=dag,
)

task_prepare_data_for_tf_idf = PythonOperator(
    #prepare_data_for_tf_idf
    task_id="prepare_data_for_tf_idf", python_callable=prepare_data_for_tf_idf, dag=dag
)

tf_idf_tasks = [
    make_train_operator_for_tf_idf(model_params, CONFIG["metrics"], dag)
    for model_params in CONFIG["models"]["tf_idf"]
]



# task_train_transformer_svm = PythonOperator(
#     task_id="train_transformer_svm", python_callable=train_transformer_svm, dag=dag
# )

# task_train_transformer_catboost = PythonOperator(
#     task_id="train_transformer_catboost",
#     python_callable=train_transformer_catboost,
#     dag=dag,
# )

# task_train_transformer_xgboost = PythonOperator(
#     task_id="train_transformer_xgboost",
#     python_callable=train_transformer_xgboost,
#     dag=dag,
# )

(
    task_init
    >> task_extract_data
    >> [task_prepare_data_for_transformer, task_prepare_data_for_tf_idf]
)

for task in tf_idf_tasks: 
    task_prepare_data_for_tf_idf.set_downstream(task)
# task_prepare_data_for_tf_idf.set_downstream(task_train_tf_idf_catboost)

# task_prepare_data_for_transformer.set_downstream(task_train_transformer_svm)
# task_prepare_data_for_transformer.set_downstream(task_train_transformer_catboost)
# task_prepare_data_for_transformer.set_downstream(task_train_transformer_xgboost)
