from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

from src.models.build import build_model
from src.datasets.build import build_dataloader
from src.utils.utils import get_rank


def run_inference(
    model: Union[torch.nn.Module, DDP],
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Run inference on validation dataset.

    Args:
        model: Model to evaluate.
        val_loader: Validation data loader.
        device: Device to run inference on.

    Returns:
        np.ndarray: Array of correct predictions (1 for correct, 0 for incorrect).
    """
    model.eval()
    all_correct = []

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for imgs, targets in tqdm(val_loader, desc="Inference"):
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs).argmax(dim=1)
            all_correct.append((preds == targets).cpu().numpy())

    return np.concatenate(all_correct)


def bootstrap_sem(
    arr: np.ndarray, B: int = 1000, seed: Optional[int] = None
) -> Tuple[float, np.ndarray]:
    """Compute bootstrap SEM and confidence intervals.

    Args:
        arr: Array of binary outcomes (1 for correct, 0 for incorrect).
        B: Number of bootstrap samples.
        seed: Random seed for reproducibility.

    Returns:
        Tuple[float, np.ndarray]: Standard error of the mean and bootstrap means.
    """
    rng = np.random.RandomState(seed)
    n = arr.shape[0]
    boot_means = np.empty(B, float)

    for b in range(B):
        idx = rng.randint(0, n, size=n)
        boot_means[b] = arr[idx].mean()

    return boot_means.std(ddof=1), boot_means


def gather_all_correct(local_correct: np.ndarray, device: torch.device) -> np.ndarray:
    """Gather correct predictions from all processes.

    Args:
        local_correct: Local array of correct predictions.
        device: Device to perform gathering on.

    Returns:
        np.ndarray: Concatenated array of correct predictions from all processes.
    """
    # Convert to tensor
    local_tensor = torch.from_numpy(local_correct.astype(np.uint8)).to(device)
    local_size = torch.tensor([local_tensor.numel()], device=device)

    # Gather sizes from all processes
    sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, local_size)
    max_size = max([s.item() for s in sizes])

    # Pad and gather tensors
    padded = torch.zeros(max_size, dtype=torch.uint8, device=device)
    padded[: local_tensor.numel()] = local_tensor
    gathered = [torch.zeros_like(padded) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, padded)

    # Concatenate results
    all_correct = []
    for g, s in zip(gathered, sizes):
        all_correct.append(g[: s.item()].cpu().numpy())
    return np.concatenate(all_correct)


def resolve_dict_errors(checkpoint: Dict) -> Dict:
    """Resolve common state dict key errors.

    Args:
        checkpoint: Model checkpoint dictionary.

    Returns:
        Dict: Cleaned state dictionary.
    """
    new_state_dict = {}

    for key, value in checkpoint.items():
        # Remove common prefixes
        if key.startswith("module."):
            key = key.replace("module.", "")
        if key.startswith("_orig_mod."):
            key = key.replace("_orig_mod.", "")

        # Skip inv_freq keys as they can be recomputed
        if key.endswith(".inv_freq"):
            continue

        new_state_dict[key] = value

    return new_state_dict


def run_sem(config: Dict) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """Run SEM evaluation on a trained model.

    Args:
        config: Configuration dictionary containing model and evaluation parameters.

    Returns:
        Tuple[Optional[float], Optional[np.ndarray]]: SEM and bootstrap means
            (only on rank 0 in distributed mode).

    Raises:
        AssertionError: If no checkpoint path is provided.
    """
    local_rank = get_rank()
    device = torch.device(f"cuda:{local_rank}")

    # Build model and load weights
    model = build_model(config)
    assert config["resume"] is not None, "No checkpoint path provided"

    checkpoint_path = config["resume"]
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(resolve_dict_errors(checkpoint["state_dict"]), strict=False)
    model.to(device)

    # Wrap model in DDP if distributed
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank])

    # Build validation dataloader
    val_loader = build_dataloader(config["validation"], True)

    # Run inference
    all_correct_local = run_inference(model, val_loader, device)

    # Gather results from all processes
    all_correct = gather_all_correct(all_correct_local, device)

    # Compute and print SEM on rank 0
    if not dist.is_initialized() or dist.get_rank() == 0:
        sem_boot, boot_means = bootstrap_sem(
            all_correct, B=config.get("bootstrap_B", 2000), seed=config.get("seed", 42)
        )
        print(f"Bootstrap SEM = {sem_boot*100:.3f} percentage points")
        ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
        print(f"95% CI = [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")
        return sem_boot, boot_means

    return None, None
