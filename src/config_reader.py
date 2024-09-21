from hydra import compose, initialize_config_dir  # , GlobalHydra
from omegaconf import OmegaConf, DictConfig
import os


def read_configs():
    # Ensure the PROJECT_ROOT environment variable is set
    project_root = os.environ.get("PROJECT_ROOT")

    # Construct the path to the config directory
    config_path = os.path.join(project_root, "configs")

    try:
        initialize_config_dir(config_dir=config_path)
    except:
        pass

    # Compose the configuration
    cfg = dict(compose(config_name="main"))

    cfg["models"] = {}
    for embedding_type in cfg["model_signatures"]:
        cfg["models"][embedding_type] = []

    merged_configs = []
    for embedding_type in cfg["model_signatures"]:
        for model in cfg["model_signatures"][embedding_type]:
            model_cfg = OmegaConf.load(
                os.path.join(config_path, embedding_type + "/" + model + ".yaml")
            )
            cfg["models"][embedding_type].append(model_cfg)
    return cfg


if __name__ == "__main__":
    print(read_configs())
