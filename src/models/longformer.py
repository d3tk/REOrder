from src.models.layers.layers import (
    ClassificationHead,
    create_patch_module,
    create_pos_emb_module,
)
import torch.nn as nn
import torch
from transformers.models.longformer.modeling_longformer import (
    LongformerEncoder,
    LongformerBaseModelOutput,
)
from typing import Optional


class ViTLongformerConfig:
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        patch_dir: str,
        pe_mode: str,
        pe_order: str,
        num_classes: int,
        custom_permute=None,
        logit_init: str | None = None,
        hidden_size=768,
        pad_token_id=0,  # Ensure pad token id is set
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=32,
        attention_window=12,
        layer_norm_eps=1e-5,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        hidden_act="gelu",
        chunk_size_feedforward=0,
        type_vocab_size=1,
    ):
        self.has_cls_token = True
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.hidden_size = hidden_size
        self.patch_dir = patch_dir
        self.pe_mode = pe_mode
        self.pe_order = pe_order
        self.custom_permute = custom_permute
        self.logit_init = logit_init
        self.num_classes = num_classes
        self.num_labels = num_classes

        # Compute max_position_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.attention_window = attention_window

        self.num_patches = (img_size // patch_size) ** 2

        if self.has_cls_token:
            self.max_position_embeddings = self.num_patches + 1
        else:
            self.max_position_embeddings = self.num_patches

        self.pad_token_id = pad_token_id  # used for positional embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_act = hidden_act
        self.chunk_size_feed_forward = chunk_size_feedforward
        self.type_vocab_size = type_vocab_size

    # need it to print as a dictionary when called for print
    def __repr__(self):
        return str(self.__dict__)


class ViTLongformer(nn.Module):
    def __init__(self, config: ViTLongformerConfig):
        super().__init__()
        self.config = config
        # Ensure attention_window is provided per layer.
        if isinstance(config.attention_window, int):
            assert (
                config.attention_window % 2 == 0
            ), "`config.attention_window` must be even"
            assert (
                config.attention_window > 0
            ), "`config.attention_window` must be positive"
            config.attention_window = [
                config.attention_window
            ] * config.num_hidden_layers
        else:
            assert (
                len(config.attention_window) == config.num_hidden_layers
            ), "`len(config.attention_window)` should equal `config.num_hidden_layers`."

        self.patch_embed = create_patch_module(
            patch_dir=config.patch_dir,
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            hidden_size=config.hidden_size,
            num_patches=config.num_patches,
            custom_permute=config.custom_permute,
            permute=False,
            logit_init=config.logit_init,
        )
        self.positional_embed = create_pos_emb_module(
            pe_order=config.pe_order,
            hidden_size=config.hidden_size,
            num_patches=config.num_patches,
            pe_mode=config.pe_mode,
            custom_permute=config.custom_permute,
            has_cls_token=True,
            prepended=True,
            pos_type="2D",
        )

        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.encoder = LongformerEncoder(config)

        self.head = ClassificationHead(
            in_size=config.hidden_size,
            out_size=config.num_classes,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _merge_to_attention_mask(
        self, attention_mask: torch.Tensor | None, global_attention_mask: torch.Tensor
    ):
        # longformer self attention expects attention mask to have 0 (no attn), 1 (local attn), 2 (global attn)
        # (global_attention_mask + 1) => 1 for local attention, 2 for global attention
        # => final attention_mask => 0 for no attention, 1 for local attention 2 for global attention
        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)  # 1 * (1+1)
        else:
            # simply use `global_attention_mask` as `attention_mask`
            # if no `attention_mask` is given
            attention_mask = global_attention_mask + 1
        return attention_mask

    def _pad_to_window_size(
        self, embeddings: torch.Tensor, attention_mask: torch.Tensor
    ):
        attention_window = (
            self.config.attention_window
            if isinstance(self.config.attention_window, int)
            else max(self.config.attention_window)
        )

        batch_size, seq_len, hidden_size = embeddings.shape
        # Calculate the number of padding tokens needed
        padding_len = (attention_window - seq_len % attention_window) % attention_window
        if padding_len > 0:
            # Create padding for embeddings: zeros
            pad_embeddings = torch.zeros(
                batch_size,
                padding_len,
                hidden_size,
                device=embeddings.device,
                dtype=embeddings.dtype,
            )
            padded_embeddings = torch.cat([embeddings, pad_embeddings], dim=1)
            # Create padding for the attention mask: zeros indicate that padded tokens are ignored
            pad_mask = torch.zeros(
                batch_size,
                padding_len,
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            padded_attention_mask = torch.cat([attention_mask, pad_mask], dim=1)
        else:
            padded_embeddings = embeddings
            padded_attention_mask = attention_mask

        return padded_embeddings, padded_attention_mask, padding_len

    def _embed(
        self, images: torch.Tensor, perm: list[int] | None = None
    ) -> torch.Tensor:

        patches = self.patch_embed(x=images, perm=perm)
        B, N, emb = patches.shape
        token_type_ids = torch.zeros(
            (B, N + 1), dtype=torch.long, device=patches.device
        )
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, hidden_size]

        # Append the CLS token
        patches = torch.cat([cls_tokens, patches], dim=1)

        # Generate position ids
        pos_ids = (
            torch.arange(N + 1, dtype=torch.long, device=patches.device)
            .unsqueeze(0)
            .expand(B, -1)
        )

        pos_embeddings = self.positional_embed(pos_seq=pos_ids)

        # permute pos_embedding to perm if perm is not none
        if perm is not None:
            if isinstance(perm, list):
                perm = torch.tensor(perm, device=pos_embeddings.device)
            perm = perm.to(dtype=torch.long, device=pos_embeddings.device)

            # shift by +1 because CLS token is prepended at position 0
            perm = perm + 1

            if perm.dim() == 1:
                perm = perm.unsqueeze(0).expand(pos_embeddings.size(0), -1)
            perm = torch.cat([torch.zeros_like(perm[..., :1]), perm], dim=-1)

            perm_exp = perm.unsqueeze(-1).expand(-1, -1, pos_embeddings.size(-1))

            pos_embeddings = torch.gather(
                pos_embeddings, 1, perm_exp
            )  # actually does the permutation

        # Combine patch embeddings, positional embeddings, and token type embeddings
        embeddings = patches + token_type_embeddings + pos_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    @torch._dynamo.disable
    def _encoder_forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
        padding_len: int,
    ) -> LongformerBaseModelOutput:
        return self.encoder(
            embeddings,
            attention_mask=attention_mask,
            padding_len=padding_len,
        )

    def forward(self, data, perm: Optional[list[int]] = None):
        # Obtain combined patch and positional embeddings.
        embeddings = self._embed(images=data, perm=perm)
        B, N, _ = embeddings.shape

        global_attention_mask = torch.zeros(
            B,
            N,
            device=embeddings.device,
            dtype=torch.long,  # Set all tokens to local attention (0)
        )
        global_attention_mask[:, 0] = 1  # set cls token to global attention (1)
        attention_mask = global_attention_mask.clone()
        embeddings, attention_mask, pad_len = self._pad_to_window_size(
            embeddings, attention_mask
        )
        attention_mask[:, N:] = -1  # set padding tokens to -1

        # Pass both masks to the encoder.
        encoder_outputs = self._encoder_forward(
            embeddings,
            padding_len=pad_len,
            attention_mask=attention_mask,
        )

        sequence_output = encoder_outputs.last_hidden_state
        logits = self.head(
            sequence_output[:, 0, :]
        )  # take input sequence without padding
        return logits


class ViTLongformer_mini(ViTLongformer):
    def __init__(self, **kwargs):
        defaults = {
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "attention_window": 50,
            "pad_token_id": 0,
        }

        merged_config = {**defaults, **kwargs}
        config = ViTLongformerConfig(**merged_config)
        super().__init__(config)


class ViTLongformer_tiny(ViTLongformer):
    def __init__(self, **kwargs):
        defaults = {
            "num_hidden_layers": 12,
            "num_attention_heads": 3,
            "hidden_size": 192,
            "intermediate_size": 768,
            "attention_window": 50,
            "pad_token_id": 0,
        }
        merged_config = {**defaults, **kwargs}
        config = ViTLongformerConfig(**merged_config)
        super().__init__(config)


class ViTLongformer_small(ViTLongformer):
    def __init__(self, **kwargs):
        defaults = {
            "num_hidden_layers": 12,
            "num_attention_heads": 6,
            "hidden_size": 384,
            "intermediate_size": 1536,
            "attention_window": 50,
            "pad_token_id": 0,
        }
        merged_config = {**defaults, **kwargs}
        config = ViTLongformerConfig(**merged_config)
        super().__init__(config)


class ViTLongformer_base(ViTLongformer):
    def __init__(self, **kwargs):
        defaults = {
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "attention_window": 50,
            "pad_token_id": 0,
        }
        merged_config = {**defaults, **kwargs}
        config = ViTLongformerConfig(**merged_config)
        super().__init__(config)


class ViTLongformer_large(ViTLongformer):
    def __init__(self, **kwargs):
        defaults = {
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "attention_window": 50,
            "pad_token_id": 0,
        }
        merged_config = {**defaults, **kwargs}
        config = ViTLongformerConfig(**merged_config)
        super().__init__(config)


class ViTLongformer_huge(ViTLongformer):
    def __init__(self, **kwargs):
        defaults = {
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
            "hidden_size": 1280,
            "intermediate_size": 5120,
            "attention_window": 50,
            "pad_token_id": 0,
        }
        merged_config = {**defaults, **kwargs}
        config = ViTLongformerConfig(**merged_config)
        super().__init__(config)


class ViTLongformer_giant(ViTLongformer):
    def __init__(self, **kwargs):
        defaults = {
            "num_hidden_layers": 48,
            "num_attention_heads": 16,
            "hidden_size": 1664,
            "intermediate_size": 6656,
            "attention_window": 50,
            "pad_token_id": 0,
        }
        merged_config = {**defaults, **kwargs}
        config = ViTLongformerConfig(**merged_config)
        super().__init__(config)


__vit_longformer_sizes__ = {
    "mini": ViTLongformer_mini,  # Approximately 0.03M parameters (example)
    "tiny": ViTLongformer_tiny,  # e.g., ~6M parameters
    "small": ViTLongformer_small,  # e.g., ~24M parameters
    "base": ViTLongformer_base,  # e.g., ~93M parameters
    "large": ViTLongformer_large,  # e.g., ~329M parameters
    "huge": ViTLongformer_huge,  # e.g., ~684M parameters
    "giant": ViTLongformer_giant,  # e.g., ~1731M parameters
}
