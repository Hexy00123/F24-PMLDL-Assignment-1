import os
import sys
import pickle
import pandas as pd
from typing import NoReturn
from sklearn.model_selection import train_test_split
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

sys.path.append(f"{os.environ['PROJECT_ROOT']}/src")
from text_process import TextPreprocessor, preprocess_targets


def prepare_data_for_transformer() -> NoReturn:
    from dags_config import LOGGER, DATA_PATH, BUCKET

    LOGGER.info(f"Open S3 connection...")

    s3_hook = S3Hook("s3_connector")
    session = s3_hook.get_session("ru-central1")
    resource = session.resource("s3")

    LOGGER.info(f"Downloading raw data from S3...")
    file = s3_hook.download_file(key=DATA_PATH + "raw/dataset.pkl", bucket_name=BUCKET)
    LOGGER.info(f"Downloaded succesfully...")

    data = pd.read_pickle(file)
    X, y = data[["text"]], data["rating"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=42
    )

    # Preprocessing targets:
    y_train = y_train.apply(preprocess_targets)
    y_test = y_test.apply(preprocess_targets)

    # Clean text:
    text_preprocessor = TextPreprocessor()
    X_train["text"] = X_train["text"].apply(text_preprocessor.clean_text)
    X_test["text"] = X_test["text"].apply(text_preprocessor.clean_text)

    # Save the cleaned text to S3 (for transformer approach)
    LOGGER.info(f"Loading cleaned data to S3...")

    for name, data in zip(
        ["X_train", "X_test", "y_train", "y_test"], [X_train, X_test, y_train, y_test]
    ):
        pickle_dataset_object = pickle.dumps(data)
        resource.Object(BUCKET, DATA_PATH + f"preprocessed/transformer/{name}.pkl").put(
            Body=pickle_dataset_object
        )

    LOGGER.info(f"Data preprocessed.")


def prepare_data_for_tf_idf() -> NoReturn:
    from dags_config import LOGGER, DATA_PATH, BUCKET

    LOGGER.info(f"Open S3 connection...")

    s3_hook = S3Hook("s3_connector")
    session = s3_hook.get_session("ru-central1")
    resource = session.resource("s3")

    LOGGER.info(f"Downloading raw data from S3...")
    file = s3_hook.download_file(key=DATA_PATH + "raw/dataset.pkl", bucket_name=BUCKET)
    LOGGER.info(f"Downloaded succesfully...")

    data = pd.read_pickle(file)
    X, y = data[["text"]], data["rating"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=42
    )

    # Preprocessing targets:
    y_train = y_train.apply(preprocess_targets)
    y_test = y_test.apply(preprocess_targets)

    # Clean text:
    text_preprocessor = TextPreprocessor()
    X_train["text"] = (
        X_train["text"]
        .apply(text_preprocessor.clean_text)
        .apply(text_preprocessor.preprocess_text)
    )
    X_test["text"] = (
        X_test["text"]
        .apply(text_preprocessor.clean_text)
        .apply(text_preprocessor.preprocess_text)
    )

    # Save the cleaned text to S3 (for transformer approach)
    LOGGER.info(f"Loading cleaned data to S3...")

    for name, data in zip(
        ["X_train", "X_test", "y_train", "y_test"], [X_train, X_test, y_train, y_test]
    ):
        pickle_dataset_object = pickle.dumps(data)
        resource.Object(BUCKET, DATA_PATH + f"preprocessed/tf_idf/{name}.pkl").put(
            Body=pickle_dataset_object
        )

    LOGGER.info(f"Data preprocessed.")
