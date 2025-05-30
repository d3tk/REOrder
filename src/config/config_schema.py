# config_schema.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from omegaconf import II
from enum import Enum


class Order(str, Enum):
    ROW = "row-major"
    COLUMN = "column-major"
    HILBERT = "hilbert-curve"
    SPIRAL = "spiral-curve"
    PEANO = "peano-curve"
    DIAGONAL = "diagonal"
    SNAKE = "snake"
    RL = "rl"
    RANDOM = "random"
    CUSTOM = "custom"


@dataclass
class Transform:
    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    name: str
    path: Optional[str] = None
    subset: float = 1.0  # Artificially cut dataset by fraction
    split: Optional[List[str]] = field(default_factory=list)
    data_transforms: List[Transform] = field(default_factory=list)
    target_transforms: Optional[Any] = None


@dataclass
class TrainingConfig:
    compile: bool
    dataset: DatasetConfig
    batch_size: int
    num_workers: int
    pin_memory: bool
    dist: bool
    shuffle: bool
    drop_last: bool
    replacement: bool
    persistent_workers: bool
    seed: int
    clip_grad: float
    num_epochs: int = 100
    num_samples: Optional[int] = None
    finetune: bool = False


@dataclass
class ValidationConfig:
    dataset: DatasetConfig
    batch_size: int
    num_workers: int
    pin_memory: bool
    dist: bool
    shuffle: bool = False
    drop_last: bool = False
    replacement: bool = False
    persistent_workers: bool = False
    seed: int = II("training.seed")  # Use the same seed as training
    num_samples: Optional[int] = None


@dataclass
class OptimizerConfig:
    name: str
    base_lr: float
    weight_decay: float
    betas: List[float]
    grad_scale: bool
    init_scale: int
    autocast: bool


@dataclass
class SchedulerConfig:
    warmup_epochs: int
    warmup_factor: int
    name: str
    min_lr_ratio: float


@dataclass
class ReinforceConfig:
    policy_optimizer: OptimizerConfig
    policy_weight_scheduler: Optional[Dict[str, float]]
    policy_gumbel_temp_scheduler: Dict[str, Union[float, int, str]]
    momentum: float
    method: str
    reward_type: str
    reward_granularity: str = "batch"
    start_after: int = 0
    logit_init: Order = Order.ROW


@dataclass
class BaseModelConfig:
    ## INP + OUTPUT PARAMS
    img_size: int
    patch_size: int
    in_chans: int
    num_classes: int

    ## PATCHER + POS EMB PARAMS
    patch_dir: Order
    pe_order: Order
    pe_mode: str
    custom_permute: Optional[List[int]] = None


@dataclass
class TXLViTConfig(BaseModelConfig):
    dropout: Optional[float] = 0.1
    mem_len: Optional[int] = 0
    clamp_len: Optional[int] = 0
    attn_type: Optional[int] = 0


@dataclass
class LongformerConfig(BaseModelConfig):
    attention_window: Optional[int] = 14


@dataclass
class Mamba2Config(BaseModelConfig):
    pass


@dataclass
class LogRules:
    project: str
    entity: str
    log: bool
    log_interval: int


@dataclass
class MainConfig:
    run_name: str
    log_rules: LogRules
    model_type: str
    size: str
    model: BaseModelConfig
    training: TrainingConfig
    validation: ValidationConfig
    optimizer: OptimizerConfig
    save_every: int
    checkpoint_path: str
    error_log_dir: str
    scheduler: Optional[SchedulerConfig] = None
    reinforce: Optional[ReinforceConfig] = None
    resume: Optional[str] = None
    slurm_id: Optional[str] = None
