import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional


def log_policy_visuals(
    permutation: torch.Tensor,
    logits: torch.Tensor,
    gumbel_noise: Optional[torch.Tensor],
    gumbel_temp: float,
    epoch: int,
    device: torch.device,
    img_size: int = 224,
    patch_size: int = 16,
) -> plt.Figure:
    """Create visualizations of policy network outputs for Weights & Biases logging.

    This function generates a figure with four subplots:
    1. Policy logits heatmap showing the raw logit values for each patch
    2. Sorted policy logits bar chart showing the distribution of logit values
    3. Stacked bar plot showing logits and Gumbel noise components
    4. Permutation rank heatmap showing the new positions of patches

    Args:
        permutation (torch.Tensor): Tensor of shape [batch_size, num_patches] or [num_patches]
            containing the permutation indices
        logits (torch.Tensor): Tensor of shape [batch_size, num_patches] or [num_patches]
            containing the policy logits
        gumbel_noise (Optional[torch.Tensor]): Optional tensor of shape [batch_size, num_patches]
            or [num_patches] containing Gumbel noise values
        gumbel_temp (float): Temperature parameter for Gumbel noise
        epoch (int): Current training epoch
        device (torch.device): Device to use for tensor operations
        img_size (int): Size of input images (default: 224)
        patch_size (int): Size of image patches (default: 16)

    Returns:
        plt.Figure: Matplotlib figure containing the policy visualizations

    Raises:
        ValueError: If gumbel_noise has unexpected dimensions
    """
    num_patches = (img_size // patch_size) ** 2
    side = img_size // patch_size
    expected_shape_1d = (num_patches,)

    # Handle batch dimension in permutation
    if permutation.ndim == 2:
        permutation_1d = permutation[0]
    elif permutation.ndim == 1:
        permutation_1d = permutation

    # Handle batch dimension in Gumbel noise
    gumbel_noise_1d = None
    if gumbel_noise is not None:
        if gumbel_noise.ndim == 2:
            gumbel_noise_1d = gumbel_noise[0]
        elif gumbel_noise.ndim == 1:
            gumbel_noise_1d = gumbel_noise
        else:
            raise ValueError(
                f"Gumbel noise has unexpected ndim: {gumbel_noise.ndim}. Expected 1 or 2 (or None)."
            )

    # Convert tensors to numpy arrays
    permutation_np = permutation_1d.detach().cpu().numpy()
    logits_np = logits.detach().cpu().numpy()

    if gumbel_noise_1d is None:
        gumbel_noise_np = np.zeros(num_patches, dtype=float)
    else:
        gumbel_noise_np = gumbel_noise_1d.detach().cpu().numpy()

    # 1. Permutation heatmap data
    new_position_array = np.zeros(num_patches, dtype=np.int32)
    for new_idx, old_idx in enumerate(permutation_np):
        new_position_array[old_idx] = (num_patches - 1) - new_idx
    new_position_2d = new_position_array.reshape(side, side)

    # 2. Sorted logits bar chart data
    sorted_inds = np.argsort(-logits_np)
    sorted_logits_np = logits_np[sorted_inds]

    # 3. Logits heatmap data
    logits_2d = logits_np.reshape(side, side)

    # 4. Stacked bar plot data
    noise_component_np = gumbel_temp * gumbel_noise_np[sorted_inds]
    sorted_logits_for_stack = np.asarray(sorted_logits_np, dtype=float)
    noise_component_for_stack = np.asarray(noise_component_np, dtype=float)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # Plot 1: Policy Logits Heatmap
    im0 = axes[0].imshow(logits_2d, cmap="viridis", aspect="equal")
    axes[0].set_title(f"Policy Logits Heatmap (Ep {epoch})")
    axes[0].set_xlabel("Patch X index")
    axes[0].set_ylabel("Patch Y index")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot 2: Sorted Policy Logits
    axes[1].bar(np.arange(num_patches), sorted_logits_np)
    axes[1].set_title(f"Sorted Policy Logits (Ep {epoch})")
    axes[1].set_xlabel("Patch rank (sorted by logit)")
    axes[1].set_ylabel("Logit value")
    axes[1].tick_params(axis="x", labelsize=8)

    # Plot 3: Logits + Gumbel Noise
    axes[2].bar(range(num_patches), sorted_logits_for_stack, label="Logit value")
    if np.any(noise_component_for_stack):
        axes[2].bar(
            range(num_patches),
            noise_component_for_stack,
            bottom=sorted_logits_for_stack,
            label=f"Noise (T={gumbel_temp:.2f})",
        )
    elif gumbel_noise is not None:
        axes[2].text(
            0.5,
            0.1,
            "Zero noise value",
            ha="center",
            va="center",
            color="gray",
            transform=axes[2].transAxes,
        )
    axes[2].set_title(f"Logits + Gumbel Noise (Ep {epoch})")
    axes[2].set_xlabel("Patch rank (sorted by logit)")
    axes[2].set_ylabel("Value")
    axes[2].legend()
    axes[2].tick_params(axis="x", labelsize=8)

    # Plot 4: Permutation Rank Heatmap
    im3 = axes[3].imshow(new_position_2d, cmap="viridis", aspect="equal")
    axes[3].set_title(f"Permutation Rank Heatmap (Ep {epoch})")
    axes[3].set_xlabel("Original Patch X index")
    axes[3].set_ylabel("Original Patch Y index")
    if num_patches <= 100:
        for r in range(side):
            for c_ in range(side):
                rank_val = int(new_position_2d[r, c_])
                text_color = "white" if rank_val < num_patches * 0.6 else "black"
                axes[3].text(
                    c_,
                    r,
                    str(rank_val),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=6,
                )
    fig.colorbar(
        im3,
        ax=axes[3],
        label="New Position Rank (Higher=Earlier)",
        fraction=0.046,
        pad=0.04,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(f"Policy Visualization - Epoch {epoch}", fontsize=16)

    return fig
