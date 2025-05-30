from functools import partial

import torch
from timm.models.vision_transformer import VisionTransformer
from timm.models._manipulate import checkpoint_seq
from .layers.layers import (
    create_patch_module,
    create_pos_emb_module,
)


class ViT(VisionTransformer):
    """Overwrite pos_emb of timm vit to allow shuffling of positional embedding"""

    def __init__(
        self,
        pe_order: str = "row-major",
        patch_dir: str = "row-major",
        pe_mode: str = "static",
        custom_permute: list[int] | None = None,
        logit_init: str | None = None,
        **kwargs,
    ):
        self.num_patches = (kwargs["img_size"] // kwargs["patch_size"]) ** 2
        self.pe_order = pe_order
        kwargs["no_embed_class"] = True
        super().__init__(
            embed_layer=partial(
                create_patch_module,
                patch_dir=patch_dir,
                img_size=kwargs["img_size"],
                patch_size=kwargs["patch_size"],
                in_chans=kwargs["in_chans"],
                hidden_size=kwargs["embed_dim"],
                num_patches=self.num_patches,
                custom_permute=custom_permute,
                permute=False,
                logit_init=logit_init,
            ),
            **kwargs,
        )
        self.pos_embed_module = create_pos_emb_module(
            pe_order=pe_order,
            hidden_size=kwargs["embed_dim"],
            num_patches=self.num_patches,
            pe_mode=pe_mode,
            custom_permute=custom_permute,
            has_cls_token=True,
            prepended=True,
            pos_type="2D",
        )
        self.pos_embed = None

    def _pos_embed(
        self, x: torch.Tensor, perm: list[int] | None = None
    ) -> torch.Tensor:
        pos_seq = torch.arange(self.num_patches, -1, -1.0, device=x.device)
        pos_embed = self.pos_embed_module(pos_seq, x.shape[0])
        pos_embed = pos_embed.permute(1, 0, 2)

        if perm is not None:
            if isinstance(perm, list):
                perm = torch.tensor(perm, device=pos_embed.device)
            perm = perm.to(dtype=torch.long, device=pos_embed.device)

            # shift by +1 because CLS token is prepended at position 0
            perm = perm + 1

            if perm.dim() == 1:
                perm = perm.unsqueeze(0).expand(pos_embed.size(0), -1)
            perm = torch.cat([torch.zeros_like(perm[..., :1]), perm], dim=-1)

            perm_exp = perm.unsqueeze(-1).expand(-1, -1, pos_embed.size(-1))

            pos_embed = torch.gather(
                pos_embed, 1, perm_exp
            )  # actually does the permutation

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        # original timm, JAX, and deit vit impls
        # pos_embed has entry for class token, concat then add
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)
        x = x + pos_embed

        return self.pos_drop(x)

    def forward(
        self, data: torch.Tensor, perm: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        # Patch embedding with optional permutation
        patches = self.patch_embed(data, perm=perm)
        embeddings = self._pos_embed(patches, perm=perm)
        embeddings = self.patch_drop(embeddings)
        norm_embeddings = self.norm_pre(embeddings)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            out = checkpoint_seq(self.blocks, norm_embeddings)
        else:
            out = self.blocks(norm_embeddings)
        out = self.norm(out)
        logits = self.forward_head(out)
        return logits


class ViT_mini(ViT):
    def __init__(self, **kwargs):
        super().__init__(embed_dim=32, depth=1, num_heads=4, **kwargs)


class ViT_T(ViT):
    def __init__(self, **kwargs):
        super().__init__(embed_dim=192, depth=12, num_heads=3, **kwargs)


class ViT_S(ViT):
    def __init__(self, **kwargs):
        super().__init__(embed_dim=384, depth=12, num_heads=6, **kwargs)


class ViT_B(ViT):
    def __init__(self, **kwargs):
        super().__init__(embed_dim=768, depth=12, num_heads=12, **kwargs)


class ViT_L(ViT):
    def __init__(self, **kwargs):
        super().__init__(embed_dim=1024, depth=24, num_heads=16, **kwargs)


class ViT_H(ViT):
    def __init__(self, **kwargs):
        super().__init__(embed_dim=1280, depth=32, num_heads=16, **kwargs)


class ViT_G(ViT):
    def __init__(self, **kwargs):
        super().__init__(
            embed_dim=1408, mlp_ratio=48 / 11, depth=40, num_heads=16, **kwargs
        )


__vit_sizes__ = {
    "mini": ViT_mini,
    "tiny": ViT_T,
    "small": ViT_S,
    "base": ViT_B,
    "large": ViT_L,
    "huge": ViT_H,
    "giant": ViT_G,
}


class ViT_timm_mini(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(embed_dim=32, depth=1, num_heads=4, **kwargs)


class ViT_timm_T(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(embed_dim=192, depth=12, num_heads=3, **kwargs)


class ViT_timm_S(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(embed_dim=384, depth=12, num_heads=6, **kwargs)


class ViT_timm_B(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(embed_dim=768, depth=12, num_heads=12, **kwargs)


class ViT_timm_L(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(embed_dim=1024, depth=24, num_heads=16, **kwargs)


class ViT_timm_H(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(embed_dim=1280, depth=32, num_heads=16, **kwargs)


class ViT_timm_G(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            embed_dim=1408, mlp_ratio=48 / 11, depth=40, num_heads=16, **kwargs
        )


__vit_timm_sizes__ = {
    "mini": ViT_timm_mini,
    "tiny": ViT_timm_T,
    "small": ViT_timm_S,
    "base": ViT_timm_B,
    "large": ViT_timm_L,
    "huge": ViT_timm_H,
    "giant": ViT_timm_G,
}
