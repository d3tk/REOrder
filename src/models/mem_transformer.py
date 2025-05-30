# Implemented in https://github.com/bair-climate-initiative/xT/blob/main/xt/models/context_encoders/transformer_xl.py
# From: https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py#L495
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.layers import (
    ClassificationHead,
    create_patch_module,
    create_pos_emb_module,
)


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = self.CoreNet(inp)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head**0.5)

        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None, mems=None):
        # multihead attention
        # [hlen x bsz x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            # layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum("ibnd,jbnd->ijbn", (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, :, :, None], -float("inf"))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None], -float("inf"))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum("ijbn,jbnd->ibnd", (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head
        )

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output


class RelMultiHeadAttn(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        dropout,
        dropatt=0,
        tgt_len=None,
        ext_len=None,
        mem_len=None,
        pre_lnorm=False,
    ):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head**0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros(
                (x.size(0), qlen - 1, x.size(2), x.size(3)),
                device=x.device,
                dtype=x.dtype,
            )
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:, :, None, None]).view(
            qlen, klen, x.size(2), x.size(3)
        )

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros(
            (x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype
        )
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(
        self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None, return_attn=False
    ):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(
            qlen, bsz, self.n_head, self.d_head
        )  # qlen x bsz x n_head x d_head

        w_head_k = w_head_k.view(
            klen, bsz, self.n_head, self.d_head
        )  # qlen x bsz x n_head x d_head

        w_head_v = w_head_v.view(
            klen, bsz, self.n_head, self.d_head
        )  # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(
            rlen, self.n_head, self.d_head
        )  # qlen x n_head x d_head

        # compute attention score
        rw_head_q = w_head_q + r_w_bias  # qlen x bsz x n_head x d_head
        AC = torch.einsum(
            "ibnd,jbnd->ijbn", (rw_head_q, w_head_k)
        )  # qlen x klen x bsz x n_head
        rr_head_q = w_head_q + r_r_bias

        BD = torch.einsum(
            "ibnd,jnd->ijbn", (rr_head_q, r_head_k)
        )  # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)
        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = (
                    attn_score.float()
                    .masked_fill(attn_mask[None, :, :, None], -float("inf"))
                    .type_as(attn_score)
                )

            elif attn_mask.dim() == 3:
                attn_score = (
                    attn_score.float()
                    .masked_fill(attn_mask[:, :, :, None], -float("inf"))
                    .type_as(attn_score)
                )

        attn_prob = F.softmax(attn_score, dim=1)

        attn_prob = self.dropatt(attn_prob)

        # compute attention vector
        attn_vec = torch.einsum("ijbn,jbnd->ibnd", (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head
        )

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = w + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        if return_attn:
            return output, attn_prob

        return output


class RelLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None):
        # r_emb: [klen, n_head, d_head], used for term B
        # r_w_bias: [n_head, d_head], used for term C
        # r_bias: [klen, n_head], used for term D

        qlen, bsz = w.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        if klen > r_emb.size(0):
            r_emb_pad = r_emb[0:1].expand(klen - r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen - r_bias.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]

        # compute attention score
        rw_head_q = w_head_q + r_w_bias[None]  # qlen x bsz x n_head x d_head

        AC = torch.einsum(
            "ibnd,jbnd->ijbn", (rw_head_q, w_head_k)
        )  # qlen x klen x bsz x n_head
        B_ = torch.einsum(
            "ibnd,jnd->ijbn", (w_head_q, r_emb)
        )  # qlen x klen x bsz x n_head
        D_ = r_bias[None, :, None]  # 1    x klen x 1   x n_head
        BD = self._rel_shift(B_ + D_)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, :, :, None], -float("inf"))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None], -float("inf"))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # compute attention vector
        attn_vec = torch.einsum("ijbn,jbnd->ibnd", (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head
        )

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = w + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(DecoderLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(
            d_model, d_inner, dropout, pre_lnorm=kwargs.get("pre_lnorm")
        )

    def forward(self, dec_inp, dec_attn_mask=None, mems=None):
        output = self.dec_attn(dec_inp, attn_mask=dec_attn_mask, mems=mems)
        output = self.pos_ff(output)

        return output


class RelLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(RelLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, **kwargs
        )
        self.pos_ff = PositionwiseFF(
            d_model, d_inner, dropout, pre_lnorm=kwargs.get("pre_lnorm")
        )

    def forward(self, dec_inp, r_emb, r_w_bias, r_bias, dec_attn_mask=None, mems=None):
        output = self.dec_attn(
            dec_inp, r_emb, r_w_bias, r_bias, attn_mask=dec_attn_mask, mems=mems
        )
        output = self.pos_ff(output)

        return output


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, **kwargs
        )
        self.pos_ff = PositionwiseFF(
            d_model, d_inner, dropout, pre_lnorm=kwargs.get("pre_lnorm")
        )

    def forward(
        self,
        dec_inp,
        r,
        r_w_bias,
        r_r_bias,
        dec_attn_mask=None,
        mems=None,
        return_attn=False,
    ):
        if return_attn:
            output, attn = self.dec_attn(
                dec_inp,
                r,
                r_w_bias,
                r_r_bias,
                attn_mask=dec_attn_mask,
                mems=mems,
                return_attn=return_attn,
            )
        else:
            output = self.dec_attn(
                dec_inp,
                r,
                r_w_bias,
                r_r_bias,
                attn_mask=dec_attn_mask,
                mems=mems,
                return_attn=return_attn,
            )
        output = self.pos_ff(output)

        if return_attn:
            return output, attn

        return output


class AdaptiveEmbedding(nn.Module):
    def __init__(
        self, n_token, d_embed, d_proj, cutoffs, div_val=1, sample_softmax=False
    ):
        super(AdaptiveEmbedding, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj

        self.emb_scale = d_proj**0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(
                nn.Embedding(n_token, d_embed, sparse=sample_softmax > 0)
            )
            if d_proj != d_embed:
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // (div_val**i)
                self.emb_layers.append(nn.Embedding(r_idx - l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_emb_i)))

    def forward(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed = F.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            emb_flat = torch.zeros(
                [inp_flat.size(0), self.d_proj],
                dtype=param.dtype,
                device=param.device,
            )
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = F.linear(emb_i, self.emb_projs[i])

                emb_flat.index_copy_(0, indices_i, emb_i)

            embed = emb_flat.view(*inp.size(), self.d_proj)

        embed.mul_(self.emb_scale)

        return embed


class MemTransformerLM(nn.Module):
    def __init__(
        self,
        # PARAMS FOR ViT
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        # PARAMS FOR PATCH ORDER PROJECT
        patch_dir="row-major",
        pe_order: str = "row-major",
        pe_mode: str = "static",
        custom_permute=None,
        logit_init=None,
        # PARAM TO SWITCH TO LLM MODE
        input_dtype="img",
        # PARAMS FOR MEM LM, DEFAULTS added
        n_layer=6,
        n_head=8,
        d_model=512,
        d_head=None,
        d_inner=2048,
        dropout=0.1,
        dropatt=0.0,
        same_length=False,
        attn_type=0,
        clamp_len=-1,
        no_memory=False,
        pre_lnorm=False,
        tgt_len=0,
        ext_len=0,
        mem_len=0,
        sample_softmax=-1,
        d_embed=None,
        n_token=None,
        # UNUSED PARAMS
        tie_weight=True,
        div_val=1,
        tie_projs=[False],
        cutoffs=[],
        adapt_inp=False,
    ):
        super(MemTransformerLM, self).__init__()

        self.input_dtype = input_dtype

        self.n_token = n_token
        if d_head is None:
            d_head = d_model // n_head

        d_embed = d_model if d_embed is None else d_embed

        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.no_memory = no_memory

        if self.input_dtype == "text":
            self.word_emb = nn.Linear(n_token, d_embed)

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len

        self.attn_type = attn_type

        self.layers = nn.ModuleList()
        if attn_type == 0:  # the default attention
            for i in range(n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(
                        n_head=n_head,
                        d_model=d_model,
                        d_head=d_head,
                        d_inner=d_inner,
                        dropout=dropout,
                        tgt_len=tgt_len,
                        ext_len=ext_len,
                        mem_len=mem_len,
                        dropatt=dropatt,
                        pre_lnorm=pre_lnorm,
                    )
                )
        elif attn_type == 1:  # learnable embeddings
            for i in range(n_layer):
                self.layers.append(
                    RelLearnableDecoderLayer(
                        n_head=n_head,
                        d_model=d_model,
                        d_head=d_head,
                        d_inner=d_inner,
                        dropout=dropout,
                        tgt_len=tgt_len,
                        ext_len=ext_len,
                        mem_len=mem_len,
                        dropatt=dropatt,
                        pre_lnorm=pre_lnorm,
                    )
                )
        elif attn_type in [2, 3]:  # absolute embeddings
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(
                        n_head=n_head,
                        d_model=d_model,
                        d_head=d_head,
                        d_inner=d_inner,
                        dropout=dropout,
                        dropatt=dropatt,
                        pre_lnorm=pre_lnorm,
                    )
                )

        self.sample_softmax = sample_softmax
        self.same_length = same_length
        self.clamp_len = clamp_len

        ## PATCH ORDER PROJECT INIT STUFF
        self.pe_mode = pe_mode
        self.pe_order = pe_order
        self.patch_dir = patch_dir

        self.img_size = img_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.patch_size = patch_size

        patches_per_dim = img_size // patch_size
        self.num_patches = patches_per_dim * patches_per_dim
        self.custom_permute = custom_permute
        if attn_type == 2:
            assert (
                self.mem_len == 0
            ), "Absolute attention (attn_type=2) requires mem_len to be 0."

        if attn_type == 3:

            qlen = self.num_patches
            assert self.mem_len >= qlen, (
                f"For deep absolute SA (attn_type=3), mem_len ({self.mem_len}) must be "
                f"at least as large as sequence length ({qlen})."
            )

        if self.input_dtype == "img":
            self.word_emb = create_patch_module(
                patch_dir=self.patch_dir,
                img_size=self.img_size,
                patch_size=self.patch_size,
                in_chans=self.in_chans,
                hidden_size=self.d_model,
                num_patches=self.num_patches,
                custom_permute=self.custom_permute,
                permute=True,
                logit_init=logit_init,
            )
            self.coord_emb = nn.Parameter(torch.zeros(self.num_patches, self.d_model))
            nn.init.trunc_normal_(self.coord_emb, std=0.02)

            self.cls_coord = nn.Parameter(torch.zeros(1, self.d_model))
            nn.init.trunc_normal_(self.cls_coord, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_embed))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.head = ClassificationHead(
            in_size=(self.d_model),
            out_size=num_classes,
        )

        self._create_params()

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        if self.attn_type == 0:  # default attention
            self.pos_emb = create_pos_emb_module(
                pe_order=self.pe_order,
                hidden_size=self.d_model,
                num_patches=self.num_patches,
                pe_mode=self.pe_mode,
                custom_permute=self.custom_permute,
                has_cls_token=True,
                prepended=True,
                pos_type="2D",
            )

            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
            nn.init.normal_(self.r_w_bias, mean=0.0, std=0.02)
            nn.init.normal_(self.r_r_bias, mean=0.0, std=0.02)

        elif self.attn_type == 1:  # learnable
            self.r_emb = nn.Parameter(
                torch.Tensor(self.n_layer, self.max_klen, self.n_head, self.d_head)
            )
            self.r_w_bias = nn.Parameter(
                torch.Tensor(self.n_layer, self.n_head, self.d_head)
            )
            self.r_bias = nn.Parameter(
                torch.Tensor(self.n_layer, self.max_klen, self.n_head)
            )
            nn.init.normal_(self.r_emb, mean=0.0, std=0.02)
            nn.init.normal_(self.r_w_bias, mean=0.0, std=0.02)
            nn.init.constant_(self.r_bias, 0)

        elif self.attn_type == 2:  # absolute standard
            self.pos_emb = create_pos_emb_module(
                pe_order=self.pe_order,
                hidden_size=self.d_model,
                num_patches=self.num_patches,
                pe_mode=self.pe_mode,
                custom_permute=self.custom_permute,
                has_cls_token=True,
                prepended=True,
                pos_type="2D",
            )

        elif self.attn_type == 3:  # absolute deeper SA
            self.r_emb = nn.Parameter(
                torch.Tensor(self.n_layer, self.max_klen, self.n_head, self.d_head)
            )
            nn.init.normal_(self.r_emb, mean=0.0, std=0.02)

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self, x):
        if self.mem_len > 0:
            mems = []
            for i in range(self.n_layer + 1):
                empty = torch.empty(0, dtype=x.dtype, device=x.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        assert len(hids) == len(mems), "len(hids) != len(mems)"

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def _forward(self, dec_inp, mems=None, perm: Optional[torch.Tensor] = None):

        word_emb = self.word_emb(dec_inp, perm)

        if self.input_dtype == "img":
            qlen, bsz, _ = word_emb.size()
            if perm is None:
                coord = self.coord_emb.unsqueeze(1).expand(-1, bsz, -1)  # [L, B, D]
            else:
                perm = perm.to(self.coord_emb.device, non_blocking=True)
                coord = self.coord_emb[perm].transpose(0, 1).contiguous()

            word_emb = word_emb + coord.to(word_emb.dtype)
        else:
            qlen, bsz, _ = dec_inp.size()

        # Prepend CLS token
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(bsz, -1, -1).permute(1, 0, 2)
            if self.input_dtype == "img":
                cls_coord = self.cls_coord.expand(bsz, -1, -1).permute(1, 0, 2)
                cls_combined = cls_token + cls_coord
            else:
                cls_combined = cls_token
            word_emb = torch.cat([cls_combined, word_emb], dim=0)

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen

        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (
                torch.triu(all_ones, 1 + mlen) + torch.tril(all_ones, -mask_shift_len)
            ).bool()[
                :, :, None
            ]  # -1
        else:
            dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen), diagonal=1 + mlen
            ).bool()[:, :, None]

        # set cls token to always be visible
        if self.cls_token is not None:
            # (CLS as query sees all keys)
            zero_row = torch.zeros(
                1, klen, 1, dtype=dec_attn_mask.dtype, device=dec_attn_mask.device
            )

            #  (All queries see CLS as key)
            zero_col = torch.zeros(
                qlen + 1, 1, 1, dtype=dec_attn_mask.dtype, device=dec_attn_mask.device
            )

            mask_with_row = torch.cat([zero_row, dec_attn_mask], dim=0)
            dec_attn_mask = torch.cat([zero_col, mask_with_row], dim=1)

        hids = []

        if self.attn_type == 0:  # default
            pos_seq = torch.arange(klen, -1, -1.0, device=word_emb.device)

            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)

            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb)
            pos_emb = self.drop(pos_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                core_out = layer(
                    core_out,
                    pos_emb,
                    self.r_w_bias,
                    self.r_r_bias,
                    dec_attn_mask=dec_attn_mask,
                    mems=mems_i,
                )
                hids.append(core_out)

        elif self.attn_type == 1:  # learnable
            core_out = self.drop(word_emb)
            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                if self.clamp_len > 0:
                    r_emb = self.r_emb[i][-self.clamp_len :]
                    r_bias = self.r_bias[i][-self.clamp_len :]
                else:
                    r_emb, r_bias = self.r_emb[i], self.r_bias[i]

                mems_i = None if mems is None else mems[i]
                core_out = layer(
                    core_out,
                    r_emb,
                    self.r_w_bias[i],
                    r_bias,
                    dec_attn_mask=dec_attn_mask,
                    mems=mems_i,
                )
                hids.append(core_out)

        elif self.attn_type == 2:  # absolute
            pos_seq = torch.arange(
                klen - 1, -1, -1.0, device=word_emb.device, dtype=word_emb.dtype
            )
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb + pos_emb[-qlen:])

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and i == 0:
                    mems_i += pos_emb[:mlen]
                core_out = layer(core_out, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)

        elif self.attn_type == 3:
            core_out = self.drop(word_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and mlen > 0:
                    cur_emb = self.r_emb[i][:-qlen]
                    cur_size = cur_emb.size(0)
                    if cur_size < mlen:
                        cur_emb_pad = cur_emb[0:1].expand(mlen - cur_size, -1, -1)
                        cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                    else:
                        cur_emb = cur_emb[-mlen:]
                    mems_i += cur_emb.view(mlen, 1, -1)
                core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

                core_out = layer(core_out, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, mlen, qlen)

        return core_out, new_mems

    def forward(self, data, *mems, perm: Optional[torch.Tensor] = None):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        if not mems:
            mems = self.init_mems(data)

        hidden, new_mems = self._forward(data, mems=mems, perm=perm)
        if self.input_dtype == "img":
            pred_hid = hidden[0]
            logits = self.head(pred_hid)
            return logits
        else:
            if self.no_memory:
                new_mems = []
            if new_mems is None:
                return [logits]
            else:
                return [logits] + new_mems


class TXL_ViT_mini(MemTransformerLM):
    def __init__(self, **kwargs):
        super().__init__(n_layer=1, n_head=1, d_model=16, d_inner=32, **kwargs)


class TXL_ViT_T(MemTransformerLM):
    def __init__(self, **kwargs):
        super().__init__(n_layer=12, n_head=3, d_model=192, d_inner=768, **kwargs)


# from mmseg.registry import MODELS


# @MODELS.register_module()
class TXL_ViT_S(MemTransformerLM):
    def __init__(self, **kwargs):
        super().__init__(n_layer=12, n_head=6, d_model=384, d_inner=1536, **kwargs)


class TXL_ViT_B(MemTransformerLM):
    def __init__(self, **kwargs):
        super().__init__(n_layer=12, n_head=12, d_model=768, d_inner=3072, **kwargs)


class TXL_ViT_L(MemTransformerLM):
    def __init__(self, **kwargs):
        super().__init__(n_layer=24, n_head=16, d_model=1024, d_inner=4096, **kwargs)


class TXL_ViT_H(MemTransformerLM):
    def __init__(self, **kwargs):
        super().__init__(n_layer=32, n_head=16, d_model=1280, d_inner=5120, **kwargs)


class TXL_ViT_G(MemTransformerLM):
    def __init__(self, **kwargs):
        super().__init__(n_layer=48, n_head=16, d_model=1664, d_inner=6656, **kwargs)


__txl_vit_sizes__ = {
    "mini": TXL_ViT_mini,  # MINI    : 0.03M parameters
    "tiny": TXL_ViT_T,  # TINY    : 6.11M parameters
    "small": TXL_ViT_S,  # SMALL   : 23.73M parameters
    "base": TXL_ViT_B,  # BASE    : 93.46M parameters
    "large": TXL_ViT_L,  # LARGE   : 329.20M parameters
    "huge": TXL_ViT_H,  # HUGE    : 684.22M parameters
    "giant": TXL_ViT_G,  # GIANT   : 1731.47M parameters
}
