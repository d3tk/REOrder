import torch.nn as nn
import torch
import math
from typing import Optional, List
from src.utils.pos_emb import get_2d_sincos_pos_embed, _get_1d_sin_cos
from src.models.layers.hilbert import (
    hilbert_scan_order,
    spiral_matrix_scan_order,
    peano_curve_scan_order,
    random_scan_order,
    diagonal_scan_bl_tr,
    snake_diagonal_scan_order,
)


def permutation_penalty(M, eps=1e-8):
    """
    Computes the matrix ℓ₁–ℓ₂ penalty on a nonnegative matrix M.
    Assumes M is of shape (N, N).
    """
    # Row penalty: for each row
    row_penalty = torch.sum(torch.abs(M), dim=1) - torch.sqrt(
        torch.sum(M**2, dim=1) + eps
    )
    # Column penalty: for each column
    col_penalty = torch.sum(torch.abs(M), dim=0) - torch.sqrt(
        torch.sum(M**2, dim=0) + eps
    )
    penalty = torch.sum(row_penalty) + torch.sum(col_penalty)
    return penalty


def threshold_and_normalize_matrix(P, eps=1e-8):
    """
    Enforces non-negativity and approximate doubly stochasticity on matrix M.
    Performs thresholding (to zero out negatives), then normalizes columns and rows.
    """
    # Threshold negatives to zero.
    P = torch.clamp(M, min=0.0)
    # Normalize columns so that each column sums to 1.
    P = P / (P.sum(dim=0, keepdim=True) + eps)
    # Normalize rows so that each row sums to 1.
    P = P / (P.sum(dim=1, keepdim=True) + eps)

    return P


class Patcher(nn.Module):
    def __init__(
        self,
        patch_size,
        img_size,
        patch_dir,
        permute_indices=None,
        logit_init=None,
    ):
        super().__init__()

        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        self.p = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.fold = nn.Fold(
            output_size=img_size, kernel_size=patch_size, stride=patch_size
        )

        self.patch_dir = patch_dir
        self.img_size = img_size
        self.h, self.w = self.img_size[0], self.img_size[1]
        self.h_patches, self.w_patches = self.h // self.p, self.w // self.p
        self.num_patches = self.h_patches * self.w_patches

        if self.patch_dir == "rl":
            # When patch_dir is RL, use logit_init for initial ordering
            if logit_init not in ["row-major"]:
                self.permute_indices = get_permute_indices(logit_init, self.num_patches)
            elif logit_init == "row-major":
                # Default case when logit_init is not specified
                self.permute_indices = None
            else:
                raise ValueError(
                    "Something is wrong. If `logit_init` is not specified, then it should be row-major by default"
                )
        elif self.patch_dir not in ["row-major"]:
            # For all other non-row-major cases (spiral, hilbert, column, etc.)
            assert (
                permute_indices is not None
            ), "permute_indices must be provided for 'column-major', 'hilbert-curve', 'spiral-curve', 'peano-curve', 'custom', or 'random' patch ordering"
            self.permute_indices = permute_indices
        else:
            # For row-major ordering
            self.permute_indices = None

    def _reorder_patches(
        self, patches: torch.Tensor, perm: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Reorder patches based on the selected order ('row-major', 'column-major', 'hilbert-curve', 'spiral-curve', 'peano-curve', 'custom').
        Args:
            patches (Tensor): Shape (B, L, C, P, P), where L = number of patches.
            perm (Tensor, optional): If provided, a 1D tensor of shape (L,) that specifies
                            the new ordering of patches, i.e. perm[i] = index of
                            the patch that goes in slot i. If None, use self.patch_dir.
        Returns:
            Tensor: Reordered patches with the same shape.
        """
        B, L, C, P1, P2 = patches.shape
        assert (
            L == self.num_patches
        ), f"Input patches L={L} does not match expected num_patches={self.num_patches}"
        device = patches.device

        if perm is not None:
            assert perm.shape == (
                B,
                L,
            ), f"Expected perm shape ({B}, {L}), got {perm.shape}"
            perm = perm.to(device, non_blocking=True)

            patches_flat = patches.view(B, L, -1)
            patch_features_dim = patches_flat.shape[-1]
            index = perm.unsqueeze(-1).expand(-1, -1, patch_features_dim)

            try:
                patches_reordered_flat = torch.gather(patches_flat, dim=1, index=index)
            except RuntimeError as e:
                print(f"Error during torch.gather in _reorder_patches:")
                print(f"  patches_flat shape: {patches_flat.shape}")
                print(f"  index shape: {index.shape}")
                print(f"  perm min/max: {perm.min()}/{perm.max()}")
                print(f"  L (dim 1 size): {L}")
                raise e

            patches = patches_reordered_flat.view(B, L, C, P1, P2).contiguous()
            return patches

        if self.patch_dir not in ["row-major"]:
            if self.permute_indices is not None:
                patches = patches[:, self.permute_indices, :, :, :]

        # For "row-major", no reordering is applied.
        return patches

    def forward(
        self, x: torch.Tensor, perm: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract patches from the input image and reorder them based on patch_dir.
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)
            perm (Tensor, optional): If provided, a permutation to reorder patches.
        Returns:
            Tensor: Patch tensor of shape (B, num_patches, C, P, P)
        """

        bs, c, h, w = x.shape
        assert (
            h % self.p == 0 and w % self.p == 0
        ), "Image dimensions must be divisible by patch size"

        x = self.unfold(x)
        patches = x.view(bs, c, self.p, self.p, -1).permute(0, 4, 1, 2, 3)
        patches = self._reorder_patches(patches, perm=perm)
        return patches

    def _reconstruct(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct the image from patches.
        Args:
            patches (Tensor): Tensor of shape (B, L, C, P, P)
        Returns:
            Tensor: Reconstructed image of shape (B, C, H, W)
        """
        bs, num_patches, c, p, _ = patches.shape
        assert num_patches == self.h_patches * self.w_patches, "Mismatch in patch count"

        reverse_indices = torch.argsort(torch.Tensor(self.permute_indices))
        patches = patches[:, reverse_indices, :, :, :]

        unfolded_shape = (bs, c * self.p * self.p, self.h_patches * self.w_patches)
        patches = patches.permute(0, 2, 3, 4, 1).contiguous().view(*unfolded_shape)
        reconstructed = self.fold(patches)
        return reconstructed


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        patch_dir,
        bias=None,  # used by Timm but we arent
        dynamic_img_pad=None,  # used by Timm but we arent
        permute_indices=None,
        permute=True,
        logit_init=None,
    ):
        """
        Patch Embedding Module for Vision Transformers.
        Args:
            img_size (int or tuple): Size of the input image (default: 224).
            patch_size (int): Size of each patch (default: 16).
            in_chans (int): Number of input channels (default: 3).
            embed_dim (int): Embedding dimension for patches (default: 768).
            patch_dir (str): patch_dir of patch extraction ('row-major', 'column-major', 'hilbert-curve', 'custom', 'spiral-curve', 'peano-curve', 'learned') (default: 'row-major')
            permute (bool): whether or not final shape is [B,N, embed_dim] (false) or [N,B,C] (True)
            logit_init (str): When patch_dir is RL, this determines the initial ordering
        """
        super().__init__()

        # Handle image size
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.patch_dim = in_chans * patch_size * patch_size
        self.permute = permute
        # Patchify
        self.patchify = Patcher(
            patch_size=patch_size,
            img_size=img_size,
            patch_dir=patch_dir,
            permute_indices=permute_indices,
            logit_init=logit_init,
        )

        # Patch Embedding: Normalize and Project Patches
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(self.patch_dim),  # Normalize flattened patches
            nn.Linear(self.patch_dim, embed_dim),  # Project to embedding dimension
            nn.LayerNorm(embed_dim),  # Normalize embeddings
        )

    def forward(
        self, x: torch.Tensor, perm: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            Tensor of shape (B, num_patches, embed_dim).
        """
        patches = self.patchify(
            x, perm=perm
        )  # Shape: (B, num_patches, C, patch_size, patch_size)
        B, num_patches, C, P1, P2 = patches.shape

        patches = patches.view(B, num_patches, -1)  # Flatten

        # Normalize and project patches
        embeddings = self.to_patch_embedding(patches)

        if self.permute:
            embeddings = embeddings.permute(1, 0, 2).contiguous()

        return embeddings


class ClassificationHead(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
    ):
        super().__init__()

        self.head = nn.Sequential(nn.LayerNorm(in_size), nn.Linear(in_size, out_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # expected shape is [bsz, 1, d_model/hidden_size]
        # ie the cls token is already extracted
        output = self.head(hidden_states)
        return output


class PositionalEmbedding(nn.Module):
    """
    Parameters:
        demb (int): Dimensionality of the embedding.
        max_position (int): Total number of positions (including a class token if used).
        mode (str): 'static' for sinusoidal embeddings or 'learned' for parameterized embeddings.
        pos_type (str): '1D' or '2D' selects the type of positional embedding.
        has_cls_token (bool): If True and pos_type=='2D', reserves the first token as a cls token (set to zeros).
        pe_order (str): Positional ordering to use (e.g. 'row-major', 'hilbert-curve', 'custom',
                        'spiral-curve', 'peano-curve'). 'row-major' means no permutation.
        permute_indices (list[int] or None): When using a non-default ordering (e.g. Hilbert),
                                               these indices are applied to reorder the embeddings.
    """

    def __init__(
        self,
        demb: int,
        max_position: int,
        mode: str,  # "static" or "learned"
        pe_order: str,  # "row-major", "hilbert-curve", "custom", "spiral-curve", "peano-curve"
        permute_indices: list[int] | None = None,
        has_cls_token: bool = False,
        prepended: bool = False,
        pos_type: str = "2D",  # "1D" or "2D"
    ):
        super(PositionalEmbedding, self).__init__()
        self.demb = demb
        self.max_position = max_position
        self.mode = mode
        self.pos_type = pos_type
        self.has_cls_token = has_cls_token
        self.pe_order = pe_order
        self.permute_indices = permute_indices

        if self.pe_order != "row-major":
            if self.permute_indices is None:
                raise ValueError(
                    f"Permutation indices must be provided for the selected pe_order {self.pe_order}."
                )
            if self.has_cls_token:
                if prepended:
                    # permute is [0, i+1 for i in permute]
                    self.permute_indices = [0] + [i + 1 for i in self.permute_indices]
                elif not prepended:
                    # permute is [i for i in permute, len of permute + 1]
                    self.permute_indices = [i for i in self.permute_indices] + [
                        len(permute_indices)
                    ]

        if self.mode == "static":
            if self.pos_type == "2D":
                # Generate 2D positional embeddings, all of our models use this.
                num_tokens = (
                    self.max_position - 1 if self.has_cls_token else self.max_position
                )  # get number of tokens
                grid_size = int(math.sqrt(num_tokens))
                if grid_size * grid_size < num_tokens:
                    grid_size += 1
                pos_emb_2d = get_2d_sincos_pos_embed(self.demb, grid_size)
                if self.has_cls_token:
                    cls_token_emb = torch.zeros(1, self.demb)
                    if prepended:  # Prepended
                        pos_emb = torch.cat([cls_token_emb, pos_emb_2d], dim=0)
                    else:  # Appended
                        pos_emb = torch.cat([pos_emb_2d, cls_token_emb], dim=0)
                else:
                    pos_emb = pos_emb_2d
                self.register_buffer("pos_emb", pos_emb)
            elif self.pos_type == "1D":
                pos_emb = _get_1d_sin_cos(torch.arange(self.max_position), self.demb)
                self.register_buffer("pos_emb", pos_emb)
            else:
                raise ValueError("Invalid pos_type. Choose '1D' or '2D'.")
        elif self.mode == "learned":
            self.pos_emb = nn.Parameter(torch.zeros(self.max_position, self.demb))
            nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        else:
            raise ValueError("Invalid mode. Choose 'static' or 'learned'.")

    def forward(self, pos_seq: torch.Tensor, bsz: int | None = None) -> torch.Tensor:
        """
        Inputs:
            pos_seq (Tensor): A tensor of positional indices.
                Can be 1D ([seq_len]) or higher-dimensional.
            bsz (int or None): If provided (for a 1D pos_seq), expands the embedding along the batch dimension.
        Returns:
            Tensor: Positional embeddings of shape [seq_len, batch_size, demb].
        """
        pos_seq = pos_seq.long()
        pos_emb = self.pos_emb[pos_seq]  # shape depends on pos_seq

        # Apply permutation
        if self.pe_order != "row-major":
            if pos_seq.dim() == 1:
                pos_emb = pos_emb[self.permute_indices, :]
            else:
                pos_emb = pos_emb[:, self.permute_indices, :]

        if pos_seq.dim() == 1:
            pos_emb = pos_emb.unsqueeze(1)  # [seq_len, 1, demb]
            if bsz is not None:
                pos_emb = pos_emb.expand(-1, bsz, -1)
        else:
            if bsz is not None:
                pos_emb = pos_emb.expand(-1, bsz, -1)
        return pos_emb


def get_permute_indices(
    order_needed: str,
    num_patches: int,
    custom_permute: List[int] | None = None,
) -> List[int] | None:

    patches_per_dim = int(math.sqrt(num_patches))

    if order_needed == "hilbert-curve":
        return hilbert_scan_order(patches_per_dim, patches_per_dim)
    elif order_needed == "spiral-curve":
        return spiral_matrix_scan_order(patches_per_dim, patches_per_dim)
    elif order_needed == "peano-curve":
        return peano_curve_scan_order(patches_per_dim, patches_per_dim)
    elif order_needed == "diagonal":
        return diagonal_scan_bl_tr(patches_per_dim, patches_per_dim)
    elif order_needed == "snake":
        return snake_diagonal_scan_order(patches_per_dim, patches_per_dim)
    elif order_needed == "custom":
        return custom_permute
    elif order_needed == "learned":
        return None
    elif order_needed == "random":
        return random_scan_order(patches_per_dim, patches_per_dim)
    elif order_needed == "column-major":
        grid = int(math.sqrt(num_patches))
        return [i * grid + j for j in range(grid) for i in range(grid)]
    else:
        return None


def create_patch_module(
    patch_dir: str,
    img_size: int,
    patch_size: int,
    in_chans: int,
    hidden_size: int,
    num_patches: int,
    custom_permute: list[int] | None = None,
    permute: bool = False,
    logit_init: str | None = None,
    **kwargs,
) -> PatchEmbed:
    return PatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=hidden_size,
        patch_dir=patch_dir,
        permute_indices=get_permute_indices(patch_dir, num_patches, custom_permute),
        permute=permute,
        logit_init=logit_init,
    )


def create_pos_emb_module(
    pe_order: str,
    hidden_size: int,
    num_patches: int,
    pe_mode: str,
    custom_permute: list[int] | None = None,
    has_cls_token: bool = True,
    prepended: bool = False,
    pos_type: str = "2D",
    **kwargs,
) -> PositionalEmbedding:
    return PositionalEmbedding(
        demb=hidden_size,
        max_position=num_patches + 1 if has_cls_token else num_patches,
        pe_order=pe_order,
        mode=pe_mode,
        permute_indices=get_permute_indices(pe_order, num_patches, custom_permute),
        has_cls_token=has_cls_token,
        prepended=prepended,
        pos_type=pos_type,
    )
