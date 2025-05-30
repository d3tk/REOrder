import torch
from typing import Optional


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: int, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Generate 2D sinusoidal positional embeddings.

    This function creates a 2D grid of positional embeddings using sinusoidal functions.
    The embeddings are concatenated from x and y coordinates, each using half of the
    embedding dimension.

    Args:
        embed_dim (int): Dimension of the embedding vectors. Must be even.
        grid_size (int): Size of the 2D grid (grid_size x grid_size patches).
        device (Optional[torch.device]): Device to create tensors on.

    Returns:
        torch.Tensor: Positional embeddings of shape [grid_size^2, embed_dim].

    Raises:
        ValueError: If embed_dim is not even.
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32, device=device)
    grid_w = torch.arange(grid_size, dtype=torch.float32, device=device)
    grid = torch.stack(torch.meshgrid(grid_w, grid_h, indexing="xy"), dim=0)

    pos_x = grid[0].flatten()
    pos_y = grid[1].flatten()

    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even for 2D sin-cos embeddings.")

    half_dim = embed_dim // 2
    emb_x = _get_1d_sin_cos(pos_x, half_dim)
    emb_y = _get_1d_sin_cos(pos_y, half_dim)

    return torch.cat([emb_x, emb_y], dim=1)


def _get_1d_sin_cos(pos: torch.Tensor, embed_dim: int) -> torch.Tensor:
    """Generate 1D sinusoidal positional embeddings.

    This function creates 1D positional embeddings using sinusoidal functions with
    different frequencies. The embeddings are concatenated from sine and cosine
    components.

    Args:
        pos (torch.Tensor): 1D tensor of positions to generate embeddings for.
        embed_dim (int): Dimension of the embedding vectors. Must be even.

    Returns:
        torch.Tensor: Positional embeddings of shape [pos.shape[0], embed_dim].
    """
    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2.0)))

    out = torch.outer(pos, omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
