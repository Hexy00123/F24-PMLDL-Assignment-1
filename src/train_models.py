from config_reader import read_configs
from typing import Callable, NoReturn
import importlib
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from itertools import combinations
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import numpy as np
import pandas as pd
from airflow.operators.python import PythonOperator
import mlflow
import os


def read_data(
    LOGGER,
    DATA_PATH,
    BUCKET,
    embeddings_type,
    files=["X_train", "X_test", "y_train", "y_test"],
):
    LOGGER.info(f"Open S3 connection...")
    s3_hook = S3Hook("s3_connector")

    LOGGER.info(f"Downloading preprocessed data from S3...")
    data = {}
    for name in files:
        data[name] = s3_hook.download_file(
            key=DATA_PATH + f"preprocessed/{embeddings_type}/{name}.pkl",
            bucket_name=BUCKET,
        )

    LOGGER.info(f"Downloaded succesfully...")
    return data


def make_train_operator_for_tf_idf(params: dict, metrics: dict, dag) -> PythonOperator:
    def python_callable() -> NoReturn:
        from dags_config import LOGGER, DATA_PATH, BUCKET

        experiment_name = params["model_name"] + "_experiment"

        try:
            # Create a new MLflow Experiment
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=f"{os.environ['MLFLOW_ARTIFACT']}/{experiment_name }",
            )
        except mlflow.exceptions.MlflowException as e:
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

        data = read_data(
            LOGGER=LOGGER,
            DATA_PATH=DATA_PATH,
            BUCKET=BUCKET,
            embeddings_type="tf_idf",
            files=["X_train", "X_test", "y_train", "y_test"],
        )

        X_train, X_test, y_train, y_test = (
            np.array(pd.read_pickle(data["X_train"])).reshape(-1),
            np.array(pd.read_pickle(data["X_test"])).reshape(-1),
            np.array(pd.read_pickle(data["y_train"])).reshape(-1),
            np.array(pd.read_pickle(data["y_test"])).reshape(-1),
        )

        LOGGER.info(f"Training model...")
        model_instance = getattr(
            importlib.import_module(params["module_name"]), params["class_name"]
        )
        pipeline = Pipeline(
            steps=[
                ("embedding", TfidfVectorizer()),
                ("classifier", model_instance()),
            ]
        )
        pipeline.set_params(**params["params"])

        with mlflow.start_run(
            run_name=f"{params['model_name']}", experiment_id=experiment_id
        ) as run:
            pipeline.fit(X_train, y_train)

            mlflow.log_params(params["params"])

            LOGGER.info(f"Evaluating {params['model_name']}...")
            predictions = pipeline.predict(X_test)

            for method_instance_name in metrics["methods"]:
                method_instance = getattr(
                    importlib.import_module(metrics["module"]), method_instance_name
                )

                metric_value = method_instance(y_test, predictions)
                LOGGER.info(f"{method_instance_name} = {metric_value}")
                mlflow.log_metric(method_instance_name, metric_value)

            signature = mlflow.models.infer_signature(
                X_test,
            )
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                signature=signature,
                input_example=X_test,
                artifact_path=f"{experiment_name}",
            )
            # mlflow.sklearn.save_model(pipeline, params["model_name"])

    # return python_callable
    return PythonOperator(
        task_id=f"{params['model_name']}", python_callable=python_callable, dag=dag
    )


def make_train_operator_for_transformer(
    params: dict, metrics: dict, dag
) -> PythonOperator:
    def python_callable() -> NoReturn:
        from dags_config import LOGGER, DATA_PATH, BUCKET

        experiment_name = params["model_name"] + "_experiment"

        try:
            # Create a new MLflow Experiment
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=f"{os.environ['MLFLOW_ARTIFACT']}/{experiment_name }",
            )
        except mlflow.exceptions.MlflowException as e:
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

        data = read_data(
            LOGGER=LOGGER,
            DATA_PATH=DATA_PATH,
            BUCKET=BUCKET,
            embeddings_type="transformer",
            files=["X_train_embeddings", "X_test_embeddings", "y_train", "y_test"],
        )
        X_train, X_test, y_train, y_test = (
            pd.read_pickle(data["X_train_embeddings"]),
            pd.read_pickle(data["X_test_embeddings"]),
            np.array(pd.read_pickle(data["y_train"])).reshape(-1),
            np.array(pd.read_pickle(data["y_test"])).reshape(-1),
        )

        LOGGER.info(f"Training model...")
        model_instance = getattr(
            importlib.import_module(params["module_name"]), params["class_name"]
        )
        pipeline = Pipeline(
            steps=[
                ("classifier", model_instance()),
            ]
        )
        pipeline.set_params(**params["params"])

        with mlflow.start_run(
            run_name=f"{params['model_name']}", experiment_id=experiment_id
        ) as run:
            pipeline.fit(X_train, y_train)

            mlflow.log_params(params["params"])

            LOGGER.info(f"Evaluating {params['model_name']}...")
            predictions = pipeline.predict(X_test)

            for method_instance_name in metrics["methods"]:
                method_instance = getattr(
                    importlib.import_module(metrics["module"]), method_instance_name
                )

                metric_value = method_instance(y_test, predictions)
                LOGGER.info(f"{method_instance_name} = {metric_value}")
                mlflow.log_metric(method_instance_name, metric_value)

            signature = mlflow.models.infer_signature(
                X_test,
            )
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                signature=signature,
                input_example=X_test,
                artifact_path=f"{experiment_name}",
            )
            # mlflow.sklearn.save_model(pipeline, params["model_name"])

    # return python_callable
    return PythonOperator(
        task_id=f"{params['model_name']}", python_callable=python_callable, dag=dag
    )


if __name__ == "__main__":
    config = read_configs()
