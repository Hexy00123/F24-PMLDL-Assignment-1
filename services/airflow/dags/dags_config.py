import logging
import os
import sys

PROJECT_ROOT = os.environ["PROJECT_ROOT"]

sys.path.append(f"{PROJECT_ROOT}/src")

from config_reader import read_configs


LOGGER = logging.getLogger()
LOGGER.addHandler(logging.StreamHandler())

BUCKET = "pmldl-assignment-1"
DATA_PATH = "data/"

CONFIG = read_configs()
