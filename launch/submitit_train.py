import argparse
import os
import socket
import sys
import uuid
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pathlib import Path

import submitit
from omegaconf import OmegaConf

from main import spmd_main
from src.config.utils import load_and_build_config, get_path_config_for_hostname


def find_latest_checkpoint(checkpoint_dir: str | Path) -> str | None:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = sorted(
        checkpoint_dir.glob("*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(checkpoints[0]) if checkpoints else None


def parse_args():
    parser = argparse.ArgumentParser("Submitit job launcher for REOrder")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--nodes", default=1, type=int)
    parser.add_argument(
        "--timeout", default=2880, type=int, help="Job timeout in minutes"
    )
    parser.add_argument("--partition", default="standard", type=str)
    parser.add_argument("--account", default="", type=str)
    parser.add_argument("--comment", default="", type=str)
    parser.add_argument("--exclude", default="", type=str)
    parser.add_argument(
        "--gpu_type",
        default="sxm",
        choices=["sxm", "pcie"],
        help="Which GPU node type to request",
    )
    return parser.parse_args()


def get_shared_folder():
    base = Path("/scratch/user/reorder-jobs")
    if not base.exists():
        raise RuntimeError("Shared folder not available.")
    return base


def get_init_file():
    shared = get_shared_folder()
    shared.mkdir(parents=True, exist_ok=True)
    init = shared / f"{uuid.uuid4().hex}_init"
    if init.exists():
        init.unlink()
    return init


class SubmititTrainer:
    def __init__(self, config_path: str, gpu_type: str):
        self.gpu_type = gpu_type
        self.config_path = config_path
        self.config = None

    def __call__(self):
        import os

        os.environ["OMP_NUM_THREADS"] = "8"
        os.environ["PYTHONUNBUFFERED"] = "1"
        if self.gpu_type == "pcie":
            os.environ["NCCL_P2P_DISABLE"] = "1"

        self._setup_dist_env()
        spmd_main(self.config)

    def checkpoint(self):
        print("⏳ Requeuing job from checkpoint...")
        self.config.dist_url = get_init_file().as_uri()

        latest_ckpt = find_latest_checkpoint(self.config.checkpoint_path)
        if latest_ckpt:
            print(f"🔁 Found checkpoint to resume from: {latest_ckpt}")
            self.config.resume = latest_ckpt
        else:
            print("⚠️ No checkpoint found, resuming from scratch.")

        return submitit.helpers.DelayedSubmission(
            SubmititTrainer(self.config_path, self.gpu_type)
        )

    def _setup_dist_env(self):
        job_env = submitit.JobEnvironment()
        hostname = socket.getfqdn()
        base_dir = Path(__file__).resolve().parent.parent

        # Set up torch distributed env vars
        dist_env = submitit.helpers.TorchDistributedEnvironment().export(
            set_cuda_visible_devices=False
        )

        # Set multiprocessing start method
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError as e:
            print(f"[{job_env.global_rank}] multiprocessing already set")

        torch.cuda.set_device(job_env.local_rank)
        print(f"Set CUDA device to local rank {job_env.local_rank}")

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=job_env.num_tasks,
            rank=job_env.global_rank,
            timeout=timedelta(minutes=30),
        )

        print("Distributed process group initialized.")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}")
        print(f"MASTER_ADDR={dist_env.master_addr} PORT={dist_env.master_port}")

        self.config = load_and_build_config(self.config_path)

        # Replace run_name in output paths with job ID
        job_id = str(job_env.job_id)
        for k in ["checkpoint_path", "error_log_dir"]:
            if k in self.config and isinstance(self.config[k], str):
                pname = Path(self.config[k])
                self.config[k] = pname.parent / f"{pname.name}_{job_id}"


def main():
    args = parse_args()

    gres = {"sxm": "gpu:A100-SXM4:8", "pcie": "gpu:A100-PCIe:8"}.get(
        args.gpu_type.lower()
    )

    if not gres:
        raise ValueError(f"Unsupported GPU type: {args.gpu_type}")

    user_config = OmegaConf.load(args.config)
    run_name = user_config.get("run_name", "unnamed")

    # Submitit job output folder
    job_dir = get_shared_folder() / run_name
    executor = submitit.AutoExecutor(folder=job_dir)

    executor.update_parameters(
        timeout_min=args.timeout,
        slurm_partition=args.partition,
        slurm_account=args.account,
        slurm_comment=args.comment,
        slurm_exclude=args.exclude,
        tasks_per_node=8,
        nodes=args.nodes,
        name=run_name,
        slurm_additional_parameters={"gres": gres},
    )

    print(f"Submitting with GRES={gres} to partition={args.partition}")
    print(f"Executor folder: {executor.folder}")

    trainer = SubmititTrainer(config_path=args.config, gpu_type=args.gpu_type)
    job = executor.submit(trainer)

    print(f"✅ Job submitted! ID: {job.job_id}")


if __name__ == "__main__":
    main()
