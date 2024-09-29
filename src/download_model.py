import os
import argparse
import requests
import mlflow
import pickle


def download_model(filename, model_uri):
    # Download the model
    loaded_model = mlflow.pyfunc.load_model(model_uri).get_raw_model()

    # Save the model
    with open(
        os.path.join(os.environ["PROJECT_ROOT"], "deployment", "model", filename),
        mode="wb",
    ) as file:
        pickle.dump(loaded_model, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and save a model.")
    parser.add_argument(
        "--filename", type=str, help="The name of the file to save the model to."
    )
    parser.add_argument("--model_uri", type=str, help="The URL of the model.")

    args = parser.parse_args()

    download_model(args.filename, args.model_uri)
