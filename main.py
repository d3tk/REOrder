import argparse
import os
import sys
from pathlib import Path
import torch
import torch.distributed as dist

# Add project root to Python path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config.utils import load_and_build_config
from src.trainer import Trainer
from src.utils.utils import get_rank


def spmd_main(config: dict, sem: str = None):
    is_distributed = config["training"]["dist"]
    already_initialized = dist.is_available() and dist.is_initialized()

    # Initialize distributed training if needed
    if is_distributed and not already_initialized:
        try:
            dist.init_process_group(backend="nccl", init_method="env://")
            torch.cuda.set_device(get_rank())
            print(
                f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
                f"rank = {dist.get_rank()}, backend={dist.get_backend()}, "
                f"device = {torch.cuda.current_device()}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize distributed training: {e}")
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    try:
        trainer = Trainer(config)
        trainer.fit()
    finally:
        # Clean up distributed training
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    """Parse command line arguments and run training or SEM evaluation."""
    parser = argparse.ArgumentParser(description="Train TXL_ViT with Config")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )

    args = parser.parse_args()

    config = load_and_build_config(args.config)
    spmd_main(config)


if __name__ == "__main__":
    main()
