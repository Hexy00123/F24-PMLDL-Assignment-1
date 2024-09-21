import pandas as pd
from sklearn.model_selection import train_test_split
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from typing import NoReturn
import pickle


def read_dataset(dataset_path: str) -> pd.DataFrame:
    def parse_tskv_line(line):
        return dict(item.split("=", 1) for item in line.strip().split("\t"))

    with open(dataset_path, "r") as file:
        data = [parse_tskv_line(line) for line in file]

    data = pd.DataFrame(data)
    data = data[data["rating"] != "0."]

    sampled_df, _ = train_test_split(
        #0.15
        data, train_size=0.15, stratify=data["rating"], random_state=42
    )
    return sampled_df


def extract_data() -> NoReturn:
    from dags_config import LOGGER, PROJECT_ROOT, BUCKET, DATA_PATH

    LOGGER.info("Reading dataset file...")
    dataset = read_dataset(f"{PROJECT_ROOT}/data/raw/geo-reviews-dataset-2023.tskv")

    LOGGER.info(f"\nDataset parameters: \n\trows: {len(dataset)}")
    LOGGER.info(f"Open S3 connection...")

    s3_hook = S3Hook("s3_connector")
    session = s3_hook.get_session("ru-central1")
    resource = session.resource("s3")

    LOGGER.info(f"Loading dataset to S3...")

    pickle_dataset_object = pickle.dumps(dataset)
    resource.Object(BUCKET, DATA_PATH + "raw/dataset.pkl").put(
        Body=pickle_dataset_object
    )

    LOGGER.info(f"Raw data extracted from source.")


if __name__ == "__main__":
    dataset = read_dataset("data/raw/geo-reviews-dataset-2023.tskv")
    print(dataset)
    print(dataset["rating"])
    print(len(dataset))
