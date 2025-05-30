import os
import socket
from pathlib import Path

from omegaconf import OmegaConf

from .config_schema import (
    BaseModelConfig,
    LongformerConfig,
    MainConfig,
    TXLViTConfig,
    Mamba2Config,
)

MODEL_CONFIG_MAP = {
    "txl_vit": TXLViTConfig,
    "mem_transformer": BaseModelConfig,
    "vit": BaseModelConfig,
    "vit-timm": BaseModelConfig,
    "longformer": LongformerConfig,
    "mamba2": Mamba2Config,
}


def get_path_config_for_hostname(hostname: str, base_dir: Path) -> Path:
    if hostname.endswith("pc."):
        return base_dir / "configs" / "paths" / "example-paths.yaml"
    else:
        raise ValueError(f"Unrecognized hostname: {hostname}")


def resolve_dataset_paths(config, dataset_paths: dict):
    for split in ["training", "validation"]:
        dataset = config[split]["dataset"]
        name = dataset["name"]

        if name not in dataset_paths:
            raise ValueError(f"Unknown dataset name '{name}' in path config")

        if split not in dataset_paths[name]:
            raise ValueError(
                f"Missing '{split}' path for dataset '{name}' in path config"
            )

        if "path" not in dataset or dataset["path"] is None:
            dataset["path"] = dataset_paths[name][split]


def load_and_build_config(config_path: str) -> MainConfig:
    user_config = OmegaConf.load(config_path)

    model_type = user_config.get("model_type")
    if model_type not in MODEL_CONFIG_MAP:
        raise ValueError(f"Unknown model_type: {model_type}")

    model_cls = MODEL_CONFIG_MAP[model_type]
    user_config["model"] = OmegaConf.merge(
        OmegaConf.structured(model_cls), user_config["model"]
    )

    hostname = socket.getfqdn()
    base_dir = Path(__file__).resolve().parent.parent.parent
    path_config_file = get_path_config_for_hostname(hostname, base_dir)

    path_config_full = OmegaConf.load(path_config_file)
    dataset_paths = OmegaConf.to_container(path_config_full.datasets, resolve=True)
    path_config_full.pop("datasets")

    config = OmegaConf.merge(OmegaConf.structured(MainConfig), user_config)
    OmegaConf.resolve(config)

    config = OmegaConf.merge(config, path_config_full)
    OmegaConf.set_struct(config, False)

    resolve_dataset_paths(config, dataset_paths)
    OmegaConf.resolve(config)

    if os.environ.get("SLURM_JOB_ID"):
        config["slurm_id"] = os.environ["SLURM_JOB_ID"]

    OmegaConf.set_struct(config, True)
    return config
