from typing import Any, Dict, List, Optional, Union
import os
import random

import torch
from torch.utils.data import DataLoader, Subset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets
from torchvision.transforms import AutoAugmentPolicy
import torchvision.transforms as transforms
import torchvision.transforms.v2 as transforms_v2
import torchvision.tv_tensors as tv_tensors
from omegaconf import OmegaConf, DictConfig

from ..utils.utils import get_world_size, get_rank


# Mapping for torchvision.transforms (v1)
T_dict = {
    "Resize": transforms.Resize,
    "RandomResizedCrop": transforms.RandomResizedCrop,
    "RandomHorizontalFlip": transforms.RandomHorizontalFlip,
    "RandomVerticalFlip": transforms.RandomVerticalFlip,
    "ColorJitter": transforms.ColorJitter,
    "RandomRotation": transforms.RandomRotation,
    "RandomCrop": transforms.RandomCrop,
    "CenterCrop": transforms.CenterCrop,
    "RandomAffine": transforms.RandomAffine,
    "RandomPerspective": transforms.RandomPerspective,
    "RandomGrayscale": transforms.RandomGrayscale,
    "GaussianBlur": transforms.GaussianBlur,
    "RandomAdjustSharpness": transforms.RandomAdjustSharpness,
    "RandomAutocontrast": transforms.RandomAutocontrast,
    "RandomInvert": transforms.RandomInvert,
    "RandomEqualize": transforms.RandomEqualize,
    "ToTensor": transforms.ToTensor,
    "Normalize": transforms.Normalize,
    "RandomApply": transforms.RandomApply,
    "RandomChoice": transforms.RandomChoice,
    "AutoAugment": transforms.AutoAugment,
    "RandAugment": transforms.RandAugment,
    "TrivialAugmentWide": transforms.TrivialAugmentWide,
}

# Mapping for torchvision.transforms.v2
T_v2_dict = {
    "ToImage": transforms_v2.ToImage,
    "RandomPhotometricDistort": transforms_v2.RandomPhotometricDistort,
    "RandomZoomOut": transforms_v2.RandomZoomOut,
    "RandomIoUCrop": transforms_v2.RandomIoUCrop,
    "RandomHorizontalFlip": transforms_v2.RandomHorizontalFlip,
    "SanitizeBoundingBoxes": transforms_v2.SanitizeBoundingBoxes,
    "ToDtype": transforms_v2.ToDtype,
}


def build_transform(
    transform_configs: List[Dict[str, Any]], use_v2: bool = False
) -> Optional[Union[transforms.Compose, transforms_v2.Compose]]:
    """Build a composition of transforms from configuration.

    Args:
        transform_configs (List[Dict[str, Any]]): List of transform configurations.
            Each config should have a 'name' key and optional 'kwargs' key.
        use_v2 (bool): Whether to use torchvision.transforms.v2.

    Returns:
        Optional[Union[transforms.Compose, transforms_v2.Compose]]: Composed transforms
            or None if transform_configs is empty.

    Raises:
        ValueError: If transform name is not registered or AutoAugment policy is invalid.
        Exception: If transform instantiation fails.
    """
    if not transform_configs:
        return None

    transform_list = []
    if use_v2:
        transform_dict = T_v2_dict
        compose_fn = transforms_v2.Compose
    else:
        transform_dict = T_dict
        compose_fn = transforms.Compose

    for cfg in transform_configs:
        transform_name = cfg["name"]
        if transform_name not in transform_dict:
            raise ValueError(
                f"Transform '{transform_name}' is not registered for "
                f"{'torchvision.transforms.v2' if use_v2 else 'torchvision.transforms'}!"
            )

        transform_cls = transform_dict[transform_name]
        kwargs = cfg.get("kwargs", {})

        # Handle AutoAugment policy
        if transform_name == "AutoAugment":
            if kwargs["policy"] == "cifar":
                kwargs["policy"] = AutoAugmentPolicy.CIFAR10
            elif kwargs["policy"] == "imagenet":
                kwargs["policy"] = AutoAugmentPolicy.IMAGENET
            else:
                raise ValueError(f"Unknown AutoAugment policy: {kwargs['policy']}")

        # Convert DictConfig to dict if needed
        if isinstance(kwargs, DictConfig):
            kwargs = OmegaConf.to_container(kwargs, resolve=True)

        # Handle v2-specific kwargs
        if use_v2:
            if transform_name == "RandomZoomOut" and "fill" in kwargs:
                fill_arg = kwargs["fill"]
                if isinstance(fill_arg, DictConfig):
                    fill_arg = OmegaConf.to_container(fill_arg, resolve=True)
                    kwargs["fill"] = fill_arg
                if isinstance(fill_arg, dict) and "Image" in fill_arg:
                    fill_arg[tv_tensors.Image] = fill_arg.pop("Image")
                    if isinstance(fill_arg[tv_tensors.Image], list):
                        fill_arg[tv_tensors.Image] = tuple(fill_arg[tv_tensors.Image])

            if transform_name == "ToDtype" and "dtype" in kwargs:
                dtype_val = kwargs["dtype"]
                if isinstance(dtype_val, str):
                    try:
                        kwargs["dtype"] = getattr(torch, dtype_val)
                    except AttributeError:
                        raise ValueError(f"Unrecognized torch dtype: {dtype_val}")

        try:
            transform_obj = transform_cls(**kwargs)
        except Exception as e:
            print(f"Error when instantiating {transform_name} with kwargs {kwargs}")
            raise e

        transform_list.append(transform_obj)

    return compose_fn(transform_list)


def build_dataset(
    config: Dict[str, Any], train: bool = True
) -> torch.utils.data.Dataset:
    """Build a dataset from configuration.

    Args:
        config (Dict[str, Any]): Dataset configuration containing:
            - name: Dataset name
            - path: Path to dataset
            - subset: Subset size (0-1)
            - data_transforms: Transform configurations
            - target_transforms: Target transform configurations
        train (bool): Whether to use training set.

    Returns:
        torch.utils.data.Dataset: The constructed dataset.

    Raises:
        ValueError: If dataset name is not supported.
    """
    if config is None:
        return None

    dataset_path = config["path"]
    subset = config["subset"]

    # Build transforms
    if config["name"] == "mscoco":
        data_transforms = build_transform(config["data_transforms"], True)
        target_transforms = None
    else:
        data_transforms = build_transform(config["data_transforms"])
        target_transforms = build_transform(config["target_transforms"])

    current_dir = os.path.dirname(os.path.abspath(__file__))
    splits_dir = os.path.normpath(os.path.join(current_dir, "..", "..", "splits"))

    # Build dataset based on name
    if config["name"] == "mnist":
        dataset = datasets.MNIST(dataset_path, train=train, transform=data_transforms)
    elif config["name"] == "imagenet-1k":
        dataset = datasets.ImageFolder(dataset_path, data_transforms, target_transforms)
    elif config["name"] == "cifar100":
        dataset = datasets.CIFAR100(
            dataset_path, train=train, transform=data_transforms, download=False
        )
    elif config["name"] == "fmow":
        from .fmow_dataset import FMoWDataset

        split_file = os.path.join(
            splits_dir, "fmow_train.txt" if train else "fmow_val.txt"
        )
        dataset = FMoWDataset(
            root_dir=config["path"],
            split_file=split_file,
            return_metadata=False,
            transform=data_transforms,
        )
    else:
        raise ValueError(f"Dataset '{config['name']}' is not supported!")

    # Create subset if needed
    if subset < 1:
        total_size = len(dataset)
        subset_size = int(total_size * subset)
        indices = random.sample(range(total_size), subset_size)
        dataset = Subset(dataset, indices)

    return dataset


def build_dataloader(config: Dict[str, Any], val: bool = False) -> DataLoader:
    """Build a dataloader from configuration.

    Args:
        config (Dict[str, Any]): Dataloader configuration containing:
            - dataset: Dataset configuration
            - dist: Whether to use distributed training
            - shuffle: Whether to shuffle data
            - seed: Random seed
            - replacement: Whether to sample with replacement
            - num_samples: Number of samples to draw
            - batch_size: Batch size
            - num_workers: Number of worker processes
            - pin_memory: Whether to pin memory
            - drop_last: Whether to drop last incomplete batch
        val (bool): Whether this is for validation.

    Returns:
        DataLoader: The constructed dataloader.
    """
    dataset = build_dataset(config["dataset"], not val)

    # Create sampler
    if config["dist"]:
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=config["shuffle"],
            seed=config["seed"],
        )
    else:
        if val:
            sampler = SequentialSampler(data_source=dataset)
        else:
            sampler = RandomSampler(
                dataset,
                replacement=config["replacement"],
                num_samples=config["num_samples"],
            )

    # Create dataloader
    batch_size = config["batch_size"]

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        drop_last=config["drop_last"],
        persistent_workers=config["persistent_workers"],
    )
