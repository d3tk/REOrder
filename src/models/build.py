from .mem_transformer import MemTransformerLM


import torch.nn as nn
from timm import create_model
from .layers.plackett_luce import PlackettLucePolicy


def build_mem_transformer(
    model_config: dict, logit_init: str | None = None
) -> nn.Module:
    return MemTransformerLM(**model_config, logit_init=logit_init)


def build_txl_vit(
    model_size, model_config: dict, logit_init: str | None = None
) -> nn.Module:
    from .mem_transformer import __txl_vit_sizes__

    model = __txl_vit_sizes__[model_size]
    return model(**model_config, logit_init=logit_init)


def build_vit(
    model_size, model_config: dict, logit_init: str | None = None
) -> nn.Module:
    from .vit import __vit_sizes__

    model = __vit_sizes__[model_size]
    return model(**model_config, logit_init=logit_init)


def build_ViTLongformer(
    model_size, model_config: dict, logit_init: str | None = None
) -> nn.Module:
    from .longformer import __vit_longformer_sizes__

    model = __vit_longformer_sizes__[model_size]
    return model(**model_config, logit_init=logit_init)


def build_ViTmamba(
    model_size, model_config: dict, logit_init: str | None = None
) -> nn.Module:
    from .vision_mamba import __vision_mamba_sizes__

    model = __vision_mamba_sizes__[model_size]
    return model(**model_config, logit_init=logit_init)


def build_vit_timm(
    model_size, model_config: dict, logit_init: str | None = None
) -> nn.Module:
    try:
        return create_model(
            f"vit_{model_size}_patch{model_config['patch_size']}_{model_config['img_size']}",
            **model_config,
        )
    except Exception as e:
        from .vit import __vit_timm_sizes__

        model = __vit_timm_sizes__[model_size]
        return model(**model_config, logit_init=logit_init)


def build_model(config: dict) -> nn.Module:
    model_type = config["model_type"]

    # Get logit_init from reinforce config if it exists
    logit_init = None
    if config.get("reinforce", None) and "logit_init" in config["reinforce"]:
        logit_init = config["reinforce"]["logit_init"]

    if model_type == "mem_transformer":
        return build_mem_transformer(config["model"], logit_init=logit_init)
    elif model_type == "txl_vit":
        return build_txl_vit(config["size"], config["model"], logit_init=logit_init)
    elif model_type == "vit":
        return build_vit(config["size"], config["model"], logit_init=logit_init)
    elif model_type == "vit-timm":
        return build_vit_timm(config["size"], config["model"], logit_init=logit_init)
    elif model_type == "longformer":
        return build_ViTLongformer(
            config["size"], config["model"], logit_init=logit_init
        )
    elif model_type == "mamba2":
        return build_ViTmamba(config["size"], config["model"], logit_init=logit_init)
    else:
        raise ValueError(f"model_type: {model_type} is not supported")


def build_policy(config: dict) -> nn.Module:
    num_patches = int(
        (config["model"]["img_size"] / config["model"]["patch_size"]) ** 2
    )
    method = config["reinforce"]["method"]
    assert method in [
        "iterative",
        "gumbel",
    ], "The Plackett-Luce sampling method has to be 'iterative' or 'gumbel'."
    granularity = config["reinforce"]["reward_granularity"]
    logit_init = config["reinforce"]["logit_init"]

    return PlackettLucePolicy(
        num_patches=num_patches,
        method=method,
        granularity=granularity,
        logit_init=logit_init,
    )
