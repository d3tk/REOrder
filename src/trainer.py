import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import contextlib
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import wandb

from src.datasets.build import build_dataloader
from src.models.build import build_model, build_policy
from src.utils.utils import (
    accuracy,
    build_optimizer,
    build_policy_schedulers,
    build_scheduler,
    get_rank,
    get_world_size,
    is_main_process,
    set_seed,
)
from .utils.wandb_plots import log_policy_visuals


WANDB_OFFLINE = os.environ.get("WANDB_MODE", None) == "offline"


class Trainer:
    """Trainer class for vision transformer models.

    This class handles the training loop, validation, checkpointing, and logging
    for vision transformer models. It supports distributed training, mixed precision,
    gradient scaling, and reinforcement learning for patch ordering.

    Attributes:
        config (Dict): Configuration dictionary.
        verbose (bool): Whether this is the main process.
        local_rank (int): Local rank for distributed training.
        world_size (int): World size for distributed training.
        device (torch.device): Device to train on.
        model (nn.Module): Model to train.
        policy (Optional[nn.Module]): Policy network for patch ordering.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (torch.optim.Optimizer): Model optimizer.
        policy_optimizer (Optional[torch.optim.Optimizer]): Policy optimizer.
        scheduler (Optional[Any]): Learning rate scheduler.
        scaler (GradScaler): Gradient scaler for mixed precision.
        criterion (nn.Module): Loss function.
        val_criterion (nn.Module): Validation loss function.
        curr_epoch (int): Current epoch number.
        logging (bool): Whether to log to Weights & Biases.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize trainer.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing:
                - training: Training parameters
                - validation: Validation parameters
                - optimizer: Optimizer parameters
                - reinforce: Reinforcement learning parameters (optional)
                - log_rules: Logging parameters (optional)

        Raises:
            AssertionError: If reward type or granularity is invalid.
            FileNotFoundError: If checkpoint file doesn't exist.
        """
        self.config = config
        self.verbose = is_main_process()
        self.local_rank = get_rank()
        self.world_size = get_world_size()

        # Set up device and seed
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.local_rank)
        set_seed(config["training"]["seed"])

        # Build model
        self.model = build_model(config).to(self.device)
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"[Rank {self.local_rank}] Model parameter count: {param_count}")
        print(
            f"[Rank {self.local_rank}] Device: {self.device}, "
            f"CUDA available: {torch.cuda.is_available()}"
        )

        # Create directories
        Path(self.config["checkpoint_path"]).mkdir(exist_ok=True)
        Path(self.config["error_log_dir"]).mkdir(exist_ok=True)

        # Load checkpoint if available
        resume_epoch = 0
        checkpoint = None
        if config.get("resume", None) is not None:
            checkpoint_path = config["resume"]
            try:
                checkpoint = torch.load(
                    checkpoint_path, map_location=self.device, weights_only=False
                )
                state_dict = checkpoint["state_dict"]
                result = self.model.load_state_dict(
                    self._resolve_dict_errors(state_dict), strict=False
                )

                if self.verbose:
                    print("Missing keys:", result.missing_keys)
                    print("Unexpected keys:", result.unexpected_keys)

                if config.get("reinforce", None) or config["training"]["finetune"]:
                    resume_epoch = 0
                else:
                    resume_epoch = checkpoint["epoch"] + 1
            except Exception as e:
                raise FileNotFoundError(
                    f"Failed to load checkpoint {checkpoint_path}: {e}"
                )

        # Set up policy network if using reinforcement learning
        self.policy = None
        if self.config.get("reinforce", None):
            assert self.config["reinforce"]["reward_type"] in [
                "binary",
                "xent",
            ], "Reward type must be one of 'binary' or 'xent'."
            self.reward_type = self.config["reinforce"]["reward_type"]

            assert self.config["reinforce"]["reward_granularity"] in [
                "batch",
                "image",
            ], "Reward granularity must be one of 'batch' or 'image'."
            self.reward_granularity = self.config["reinforce"]["reward_granularity"]
            self.policy = build_policy(config).to(self.device)

            self.policy_start_after = self.config["reinforce"]["start_after"]
            self.policy_running_epochs = (
                self.config["reinforce"]
                .get("policy_gumbel_temp_scheduler", {})
                .get(
                    "running_epochs",
                    config["training"]["num_epochs"] - self.policy_start_after,
                )
            )

        # Compile model if requested
        if self.config["training"]["compile"]:
            torch._dynamo.config.optimize_ddp = True
            torch._dynamo.config.capture_scalar_outputs = True
            self.model = torch.compile(self.model, dynamic=True)
            if self.policy:
                self.policy = torch.compile(self.policy)

        # Set up distributed training
        if config["training"]["dist"]:
            self.model = DDP(
                self.model, device_ids=[self.local_rank], find_unused_parameters=False
            )
            if self.policy is not None:
                self.policy = DDP(self.policy, device_ids=[self.local_rank])
            else:
                self.policy = None

        # Build data loaders
        self.train_loader = build_dataloader(config["training"])
        self.val_loader = build_dataloader(config["validation"], True)

        # Build optimizers
        self.optimizer = build_optimizer(config["optimizer"], self.model.parameters())
        if self.policy:
            self.policy_optimizer = build_optimizer(
                config["reinforce"]["policy_optimizer"], self.policy.parameters()
            )
            self.policy_weight_scheduler, self.policy_gumbel_temp_scheduler = (
                build_policy_schedulers(
                    policy=self.policy,
                    steps_per_epoch=len(self.train_loader),
                    config=config,
                )
            )
        else:
            self.policy_optimizer = None
            self.policy_weight_scheduler = None
            self.policy_gumbel_temp_scheduler = None

        # Set up policy parameters
        self.baseline_momentum = (
            config["reinforce"]["momentum"] if self.policy else None
        )
        self.policy_weight = (
            self.policy_weight_scheduler.get_current_weight()
            if self.policy_weight_scheduler
            else 1.0
        )

        # Build scheduler
        if config["training"]:
            self.scheduler = build_scheduler(
                config, self.optimizer, len(self.train_loader)
            )
        else:
            self.scheduler = None

        # Set up gradient scaler
        self.scaler = GradScaler(
            enabled=config["optimizer"]["grad_scale"],
            init_scale=config["optimizer"]["init_scale"],
        )

        # Load checkpoint states if available
        if checkpoint is not None and not config.get("finetune", None):
            self._load_checkpoint_states(checkpoint)

        self.checkpoint_path = self.config["checkpoint_path"]
        self.curr_epoch = resume_epoch

        # Set up loss functions
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.val_criterion = nn.CrossEntropyLoss(reduction="sum").to(self.device)

        # Set up logging
        if config.get("log_rules", {}).get("log", False) and self.verbose:
            self._init_wandb()
            if WANDB_OFFLINE:
                from wandb_osh.hooks import TriggerWandbSyncHook

                self.trigger_sync = TriggerWandbSyncHook()

            self.log_interval = config["log_rules"]["log_interval"]
            self.logging = True
            wandb.watch(self.model, log="all", log_freq=self.log_interval, idx=0)
            if self.policy:
                wandb.watch(self.policy, log="all", log_freq=self.log_interval, idx=1)

            wandb.define_metric("Val Loss", summary="min")
            wandb.define_metric("Validation Top-1 Accuracy", summary="max")
            wandb.define_metric("Validation Top-5 Accuracy", summary="max")
        else:
            self.logging = False

    def _load_checkpoint_states(self, checkpoint: Dict[str, Any]) -> None:
        """Load states from checkpoint.

        Args:
            checkpoint (Dict[str, Any]): Checkpoint dictionary.
        """
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if self.policy and checkpoint.get("policy_dict", None):
            policy_to_load = (
                self.policy.module if isinstance(self.policy, DDP) else self.policy
            )
            policy_to_load.load_state_dict(checkpoint["policy_dict"])

        if self.policy_optimizer and checkpoint.get(
            "policy_optimizer_state_dict", None
        ):
            self.policy_optimizer.load_state_dict(
                checkpoint["policy_optimizer_state_dict"]
            )

        if self.policy_weight_scheduler and checkpoint.get(
            "policy_weight_scheduler_state_dict", None
        ):
            self.policy_weight_scheduler.load_state_dict(
                checkpoint["policy_weight_scheduler_state_dict"]
            )

        if self.policy_gumbel_temp_scheduler and checkpoint.get(
            "policy_gumbel_temp_scheduler_state_dict", None
        ):
            self.policy_gumbel_temp_scheduler.load_state_dict(
                checkpoint["policy_gumbel_temp_scheduler_state_dict"]
            )
            current_tau = self.policy_gumbel_temp_scheduler.get_current_tau()
            self.policy_gumbel_temp_scheduler._set_policy_tau(current_tau)

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        wandb.init(
            project=self.config["log_rules"]["project"],
            entity=self.config["log_rules"]["entity"],
            name=f"{self.config['run_name']}",
            config=OmegaConf.to_container(self.config),
            dir=self.config["checkpoint_path"],
        )

    def _policy_step(
        self,
        output: torch.Tensor,
        labels: torch.Tensor,
        log_prob_perm: torch.Tensor,
    ):
        """
        Compute reward, advantage and policy loss (including weight),
        update the running baseline inplace, and return everything needed
        for backprop and logging.
        """
        B = output.size(0)  # batch size
        reward = torch.zeros(B, device=output.device)

        # Only track gradients if the reward_type needs it (xent)
        ctx = torch.no_grad() if output.requires_grad else contextlib.nullcontext()

        with ctx:
            if self.reward_type == "xent":
                per_img_loss = torch.nn.functional.cross_entropy(
                    output.float(), labels, reduction="none"  # loss per image
                )
                reward = -per_img_loss
            elif self.reward_type == "binary":
                preds = output.argmax(dim=1)
                reward = ((preds == labels).float() * 2) - 1.0
            else:
                # Should never get here
                raise ValueError(f"Unknown reward_type: {self.reward_type}")

        policy_module = (
            self.policy.module if hasattr(self.policy, "module") else self.policy
        )
        current_baseline = policy_module.running_baseline

        advantage = torch.zeros_like(reward)
        base_update_val = 0.0

        assert (
            log_prob_perm.ndim == 1 and log_prob_perm.shape[0] == B
        ), f"Log prob must be shape [{B}] for image granularity. Got {log_prob_perm.shape}"
        advantage = reward - current_baseline
        base_update_val = reward.mean().item()

        policy_module.running_baseline.copy_(
            policy_module.running_baseline * self.baseline_momentum
            + (1.0 - self.baseline_momentum) * base_update_val
        )

        policy_loss_terms = -(advantage * log_prob_perm)
        policy_loss = policy_loss_terms.mean()

        total_policy_loss = self.policy_weight * policy_loss

        return {
            "total_policy_loss": total_policy_loss,
            "policy_loss": policy_loss.detach(),
            "reward": reward.mean().item(),
            "advantage": advantage.mean().item(),
        }

    def fit(self):
        for epoch in range(self.curr_epoch, self.config["training"]["num_epochs"]):
            try:
                # Set epoch for distributed samplers
                if self.config["training"]["dist"]:
                    self.train_loader.sampler.set_epoch(epoch)
                    if self.val_loader.sampler is not None:
                        self.val_loader.sampler.set_epoch(epoch)

                # Train for one epoch
                train_loss = self.train_one_epoch(epoch)
                if self.config["training"]["dist"]:
                    dist.barrier()

                # Validate
                val_loss, val_acc1, val_acc5 = self.validate_one_epoch(epoch)

                # Update policy parameters if using reinforcement learning
                if self.policy and epoch >= self.policy_start_after:
                    if self.policy_weight_scheduler:
                        self.policy_weight = self.policy_weight_scheduler.step()
                    if self.policy_gumbel_temp_scheduler:
                        self.policy_gumbel_temp_scheduler.step()

                # Log metrics
                if self.local_rank == 0 and self.logging:
                    wandb.log(
                        {
                            "Training Loss": train_loss,
                            "Val Loss": val_loss,
                            "Validation Top-1 Accuracy": val_acc1,
                            "Validation Top-5 Accuracy": val_acc5,
                            "Policy Weight": (
                                self.policy_weight if self.policy else None
                            ),
                            "Policy Temperature": (
                                self.policy_gumbel_temp_scheduler.get_current_tau()
                                if self.policy_gumbel_temp_scheduler
                                else None
                            ),
                            "Epoch": epoch,
                        }
                    )
                    if WANDB_OFFLINE:
                        self.trigger_sync()

                # Save checkpoint periodically
                if self.local_rank == 0 and epoch % self.config["save_every"] == 0:
                    self.save_checkpoint(epoch)

                if self.config["training"]["dist"]:
                    dist.barrier()

            except RuntimeError as e:
                if self.local_rank == 0:
                    self.save_checkpoint(epoch, "ERROR")
                raise RuntimeError(f"Training failed at epoch {epoch}: {e}")

        # Save final model
        if self.local_rank == 0:
            self.save_checkpoint(epoch)

    def train_one_epoch(self, epoch: int) -> Tuple[float, float, float]:
        """Train for one epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Tuple[float, float, float]: Average training loss, top-1 accuracy,
                and top-5 accuracy.
        """
        self.model.train()
        if self.policy:
            self.policy.train()

        # Set up progress bar and metrics
        if self.verbose:
            train_loader_tqdm = tqdm(self.train_loader, total=len(self.train_loader))
        else:
            train_loader_tqdm = self.train_loader

        # Initialize metrics
        running_loss = 0.0
        num_samples = 0

        for i, (images, labels) in enumerate(train_loader_tqdm):
            # Move data to device
            images = images.cuda(self.local_rank, non_blocking=True)
            labels = labels.cuda(self.local_rank, non_blocking=True)
            batch_size = images.size(0)

            # Get policy outputs if using reinforcement learning
            if self.policy and epoch >= self.policy_start_after:
                policy_module = (
                    self.policy.module
                    if hasattr(self.policy, "module")
                    else self.policy
                )
                is_policy_active = (
                    self.policy
                    and epoch >= self.policy_start_after
                    and epoch < (self.policy_start_after + self.policy_running_epochs)
                )
                if is_policy_active:
                    current_gumbel_temp = (
                        self.policy_gumbel_temp_scheduler.get_current_tau()
                    )
                    policy_module.temperature = current_gumbel_temp
                    dummy = torch.zeros(1, device=images.device)
                    permutations, log_prob_perms, gumbel_noise = self.policy(
                        dummy_input=dummy,
                        batch_size=batch_size,
                    )
                else:
                    with torch.no_grad():
                        policy_logits = policy_module.logits
                    det_perm = torch.argsort(policy_logits, descending=True)
                    permutations = det_perm.unsqueeze(0).repeat(batch_size, 1)
            else:
                permutations = None

            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)
            if self.policy_optimizer:
                self.policy_optimizer.zero_grad(set_to_none=True)

            # Forward pass with mixed precision
            with autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=self.config["optimizer"]["autocast"],
            ):

                output = self.model(data=images, perm=permutations)
                # print(output.shape, labels.shape)
                loss = self.criterion(output, labels)

            # Backward pass
            self.scaler.scale(loss).backward()

            # Policy step if active
            is_policy_active = (
                self.policy
                and epoch >= self.policy_start_after
                and epoch < (self.policy_start_after + self.policy_running_epochs)
            )
            if is_policy_active:
                output_for_policy = output.detach()
                metrics = self._policy_step(output_for_policy, labels, log_prob_perms)
                total_policy_loss = metrics["total_policy_loss"]
                self.scaler.scale(total_policy_loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config["training"]["clip_grad"]
            )
            if is_policy_active:
                self.scaler.unscale_(self.policy_optimizer)
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config["training"]["clip_grad"]
                )

            # Optimizer step
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            if self.policy_optimizer and is_policy_active:
                self.scaler.step(self.policy_optimizer)
            self.scaler.update()

            # Learning rate scheduling
            skip_lr_sched = scale_before > self.scaler.get_scale()
            if not skip_lr_sched and self.scheduler is not None:
                self.scheduler.step()

            # Update metrics
            running_loss += loss.item() * batch_size
            num_samples += batch_size

            # Update progress bar
            if self.verbose:
                postfix_dict = {
                    "epoch": epoch,
                    "loss": loss.item(),
                    "lr": (
                        self.scheduler.get_last_lr()[-1]
                        if self.scheduler is not None
                        else self.optimizer.param_groups[0]["lr"]
                    ),
                    "mem": f"{torch.cuda.max_memory_reserved() / 1024 ** 3:.2f}G",
                }
                if is_policy_active:
                    postfix_dict["policy_loss"] = metrics["policy_loss"].item()
                    postfix_dict["total_loss"] = metrics["total_policy_loss"].item()
                    postfix_dict["policy_weight"] = self.policy_weight

                train_loader_tqdm.set_postfix(postfix_dict)
                train_loader_tqdm.update()

                # Log to wandb
                if self.logging and i % self.log_interval == 0:
                    self._log_training_metrics(
                        epoch=epoch,
                        loss=loss,
                        policy_metrics=metrics if is_policy_active else None,
                        permutations=permutations if is_policy_active else None,
                        gumbel_noise=gumbel_noise if is_policy_active else None,
                    )

        # Return average loss
        return running_loss / num_samples

    def validate_one_epoch(self, epoch: int) -> Tuple[float, float, float]:
        """Validate for one epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Tuple[float, float, float]: Average validation loss, top-1 accuracy,
                and top-5 accuracy.
        """
        self.model.eval()
        if self.policy:
            self.policy.eval()

        # Set up progress bar
        if self.local_rank == 0:
            pbar = tqdm(self.val_loader, total=len(self.val_loader))
        else:
            pbar = self.val_loader

        # Initialize metrics
        local_loss = 0.0
        local_top1 = 0.0
        local_top5 = 0.0
        local_samples = 0

        # Get policy permutation if using reinforcement learning
        if self.policy and epoch >= self.policy_start_after:
            policy_logits = (
                self.policy.module.logits
                if hasattr(self.policy, "module")
                else self.policy.logits
            )
            val_perm = torch.argsort(policy_logits, descending=True)
        else:
            val_perm = None

        # Run validation
        with torch.no_grad(), autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=self.config["optimizer"]["autocast"],
        ):
            for images, labels in pbar:
                # Move data to device
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                batch_size = images.size(0)

                # Apply policy permutation if available
                if val_perm is not None:
                    repeated_perm = val_perm.repeat(batch_size, 1)
                else:
                    repeated_perm = None

                # Forward pass
                output = self.model(images, perm=repeated_perm)
                batch_loss = self.val_criterion(output, labels)

                # Update metrics
                acc1, acc5 = accuracy(output, labels, top_k=(1, 5))
                local_loss += batch_loss.item()
                local_top1 += acc1.item()
                local_top5 += acc5.item()
                local_samples += batch_size

                # Update progress bar
                if self.local_rank == 0:
                    pbar.set_postfix(
                        {
                            "val_loss": local_loss / local_samples,
                            "val_top1": local_top1 / local_samples,
                            "val_top5": local_top5 / local_samples,
                        }
                    )
                    pbar.update()

        # Compute average metrics
        val_loss = local_loss / local_samples
        val_top1 = local_top1 / local_samples
        val_top5 = local_top5 / local_samples

        # Gather metrics from all processes if distributed
        if (
            self.config["validation"].get("dist", False)
            and torch.distributed.is_initialized()
        ):
            sums = torch.tensor(
                [local_loss, local_top1, local_top5, local_samples],
                device=images.device,
            )
            torch.distributed.all_reduce(sums, op=torch.distributed.ReduceOp.SUM)
            total_loss, total_top1, total_top5, total_count = sums.tolist()

            val_loss = total_loss / total_count
            val_top1 = total_top1 / total_count
            val_top5 = total_top5 / total_count

            torch.distributed.barrier()
        elif torch.distributed.is_initialized():
            torch.distributed.barrier()

        return val_loss, val_top1, val_top5

    def _log_training_metrics(
        self,
        epoch: int,
        loss: torch.Tensor,
        policy_metrics: Optional[Dict[str, torch.Tensor]] = None,
        permutations: Optional[torch.Tensor] = None,
        gumbel_noise: Optional[torch.Tensor] = None,
    ) -> None:
        """Log training metrics to Weights & Biases.

        Args:
            epoch: Current epoch number.
            loss: Training loss.
            policy_metrics: Optional dictionary of policy-related metrics.
            permutations: Optional tensor of policy permutations for visualization.
            gumbel_noise: Optional tensor of Gumbel noise for visualization.
        """
        log_dict = {
            "Training Loss": loss.item(),
            "Learning Rate": (
                self.scheduler.get_last_lr()[-1]
                if self.scheduler is not None
                else self.optimizer.param_groups[0]["lr"]
            ),
            "Epoch": epoch,
        }

        if policy_metrics and permutations is not None and gumbel_noise is not None:
            log_dict.update(
                {
                    "policy_baseline": self.policy.module.running_baseline,
                    "policy_loss": policy_metrics["policy_loss"].item(),
                    "total_loss": policy_metrics["total_policy_loss"].item(),
                    "policy_weight": self.policy_weight,
                    "gumbel_temp": self.policy_gumbel_temp_scheduler.get_current_tau(),
                    "reward": (
                        policy_metrics["reward"].mean().item()
                        if isinstance(policy_metrics["reward"], torch.Tensor)
                        else policy_metrics["reward"]
                    ),
                    "advantage": (
                        policy_metrics["advantage"].mean().item()
                        if isinstance(policy_metrics["advantage"], torch.Tensor)
                        else policy_metrics["advantage"]
                    ),
                    "mean_noise_logit_magnitude_ratio": (
                        torch.linalg.vector_norm(
                            gumbel_noise
                            * self.policy_gumbel_temp_scheduler.get_current_tau()
                        )
                        / torch.linalg.vector_norm(self.policy.module.logits.detach())
                    ),
                }
            )

            # Add policy visualization
            fig = log_policy_visuals(
                permutation=permutations,
                logits=self.policy.module.logits.detach(),
                gumbel_noise=gumbel_noise,
                gumbel_temp=self.policy_gumbel_temp_scheduler.get_current_tau(),
                epoch=epoch,
                device=self.device,
                img_size=self.config["model"]["img_size"],
                patch_size=self.config["model"]["patch_size"],
            )
            log_dict["Plackett-Luce Visualization"] = wandb.Image(fig)
            plt.close(fig)

        wandb.log(log_dict)
        if WANDB_OFFLINE:
            self.trigger_sync()

    def save_checkpoint(self, epoch: int, ext: str | None = None):
        """Save the current model, optimizer, scheduler, and scaler states only on rank 0"""

        model_state_dict = (
            self.model.module.state_dict()
            if isinstance(self.model, DDP)
            else self.model.state_dict()
        )

        policy_state_dict = None
        if self.policy:
            policy_to_save = (
                self.policy.module if isinstance(self.policy, DDP) else self.policy
            )
            policy_state_dict = policy_to_save.state_dict()

        snapshot = {
            "state_dict": model_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "scaler_state_dict": self.scaler.state_dict(),
            "policy_dict": policy_state_dict,
            "policy_optimizer_state_dict": (
                self.policy_optimizer.state_dict() if self.policy_optimizer else None
            ),
            "policy_weight_scheduler_state_dict": (
                self.policy_weight_scheduler.state_dict()
                if self.policy_weight_scheduler
                else None
            ),
            "policy_gumbel_temp_scheduler_state_dict": (
                self.policy_gumbel_temp_scheduler.state_dict()
                if self.policy_gumbel_temp_scheduler
                else None
            ),
            "epoch": epoch,
        }
        if ext is None:
            fname = f"{self.config['run_name']}-ep{epoch}_rank{self.local_rank}.pt"
        else:
            fname = f"{self.config['run_name']}-ep{epoch}_ERROR.pt"

        save_path = Path(self.checkpoint_path) / fname
        torch.save(snapshot, save_path)

        latest_symlink = Path(self.checkpoint_path) / "latest.pt"
        # Remove existing symlink or file
        if latest_symlink.exists() or latest_symlink.is_symlink():
            latest_symlink.unlink()
        latest_symlink.symlink_to(save_path)

        print(f"Epoch {epoch} | Checkpoint saved at {str(save_path)}")

    def _resolve_dict_errors(self, checkpoint: dict):
        new_state_dict = {}

        for key, value in checkpoint.items():
            if key.startswith("module."):
                key = key.replace("module.", "")

            if key.startswith("_orig_mod."):
                key = key.replace("_orig_mod.", "")

            # This may not be present, so we'll just recompute it
            if key.endswith(".inv_freq"):
                continue

            new_state_dict[key] = value

        return new_state_dict
