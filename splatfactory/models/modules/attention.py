# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py


import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None, attn_mask=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if attn_mask is not None:
            attn_mask = attn_mask[:, None].repeat(1, self.num_heads, 1, 1)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop.p if self.training else 0.0
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,
        attn_mode: str = "softmax",
        separate_kv_proj: bool = False,  # If True, use separate k_proj/v_proj (for Pi3X)
        rope=None,  # RoPE for positional encoding
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn
        self.attn_mode = attn_mode
        self.separate_kv_proj = separate_kv_proj
        valid_modes = {"softmax", "sigmoid", "linear", "slot"}
        assert attn_mode in valid_modes, f"Invalid attention mode {attn_mode}"

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        if separate_kv_proj:
            # Separate k and v projections (matches Pi3X checkpoint structure)
            self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        else:
            # Combined kv projection (default, backward compatible)
            self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(
        self,
        queries: Tensor,
        context: Tensor,
        qpos=None,
        kpos=None,
        return_attention: bool = False,
    ) -> Tensor:
        B, N_q, C = queries.shape
        N_c = context.shape[1]

        q = self.q_proj(queries).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.separate_kv_proj:
            k = (
                self.k_proj(context)
                .reshape(B, N_c, self.num_heads, self.head_dim)
                .permute(0, 2, 1, 3)
            )
            v = (
                self.v_proj(context)
                .reshape(B, N_c, self.num_heads, self.head_dim)
                .permute(0, 2, 1, 3)
            )
        else:
            kv = (
                self.kv_proj(context)
                .reshape(B, N_c, 2, self.num_heads, self.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            k, v = kv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, qpos)
            k = self.rope(k, kpos)

        # Slot attention can't use fused attention
        use_fused = self.fused_attn and self.attn_mode == "softmax" and not return_attention

        if use_fused:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
            )
            attn_normalized = None
        else:
            q_scaled = q * self.scale
            attn = q_scaled @ k.transpose(-2, -1)  # [B, H, N_q, N_c]

            if self.attn_mode == "slot":
                # Slot attention: softmax over slots (queries), not context
                attn_weights = attn.softmax(dim=-2)  # [B, H, N_q, N_c] - context chooses slots
                attn_normalized = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
            elif self.attn_mode == "sigmoid":
                attn_weights = torch.sigmoid(attn)
                attn_normalized = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
            elif self.attn_mode == "linear":
                attn_weights = F.relu(attn)
                attn_normalized = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
            else:  # softmax
                attn_normalized = attn.softmax(dim=-1)

            attn_normalized = self.attn_drop(attn_normalized)
            x = attn_normalized @ v

        x = x.transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn_normalized.mean(dim=1)
        return x


class MultiSourceCrossAttention(nn.Module):
    """Per-source Q and KV, heads partitioned, then concat + Linear O.

    For ZipSplat Giant: num_heads=24, num_sources=3 → 8 heads per source.
    `proj` keeps the CrossAttention O shape so that key loads from the ckpt.
    `q_proj` / `kv_proj` are sliced from the single-source matrices on load.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_sources: int = 3,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,
        attn_mode: str = "softmax",
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        assert num_heads % num_sources == 0, (
            f"num_heads={num_heads} must be divisible by num_sources={num_sources}"
        )
        valid_modes = {"softmax", "sigmoid", "linear", "slot"}
        assert attn_mode in valid_modes, f"Invalid attention mode {attn_mode}"

        self.num_heads = num_heads
        self.num_sources = num_sources
        self.heads_per_source = num_heads // num_sources
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn
        self.attn_mode = attn_mode
        self.rope = rope

        kv_dim = self.heads_per_source * self.head_dim
        self.q_proj = nn.ModuleList(
            [nn.Linear(dim, kv_dim, bias=qkv_bias) for _ in range(num_sources)]
        )
        self.kv_proj = nn.ModuleList(
            [nn.Linear(dim, kv_dim * 2, bias=qkv_bias) for _ in range(num_sources)]
        )
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        src_dim = self.heads_per_source * self.head_dim
        old_qw = state_dict.pop(prefix + "q_proj.weight", None)
        old_qb = state_dict.pop(prefix + "q_proj.bias", None)
        if old_qw is not None:
            for i in range(self.num_sources):
                state_dict[prefix + f"q_proj.{i}.weight"] = old_qw[i * src_dim : (i + 1) * src_dim]
                if old_qb is not None:
                    state_dict[prefix + f"q_proj.{i}.bias"] = old_qb[i * src_dim : (i + 1) * src_dim]

        # Slice a single-source kv_proj [2D, D] = [K; V] into per-source [2*(D/S), D].
        old_w = state_dict.pop(prefix + "kv_proj.weight", None)
        old_b = state_dict.pop(prefix + "kv_proj.bias", None)
        if old_w is not None:
            dim = old_w.shape[0] // 2
            for i in range(self.num_sources):
                k = old_w[i * src_dim : (i + 1) * src_dim]
                v = old_w[dim + i * src_dim : dim + (i + 1) * src_dim]
                state_dict[prefix + f"kv_proj.{i}.weight"] = torch.cat([k, v], dim=0)
                if old_b is not None:
                    kb = old_b[i * src_dim : (i + 1) * src_dim]
                    vb = old_b[dim + i * src_dim : dim + (i + 1) * src_dim]
                    state_dict[prefix + f"kv_proj.{i}.bias"] = torch.cat([kb, vb], dim=0)
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _attend(self, q: Tensor, k: Tensor, v: Tensor, return_attention: bool):
        use_fused = self.fused_attn and self.attn_mode == "softmax" and not return_attention
        if use_fused:
            return (
                F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
                ),
                None,
            )

        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.attn_mode == "slot":
            attn_weights = attn.softmax(dim=-2)
            attn_normalized = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
        elif self.attn_mode == "sigmoid":
            attn_weights = torch.sigmoid(attn)
            attn_normalized = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
        elif self.attn_mode == "linear":
            attn_weights = F.relu(attn)
            attn_normalized = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            attn_normalized = attn.softmax(dim=-1)
        attn_normalized = self.attn_drop(attn_normalized)
        return attn_normalized @ v, attn_normalized

    def forward(
        self,
        queries: Tensor,
        contexts,
        qpos=None,
        kpos=None,
        return_attention: bool = False,
    ) -> Tensor:
        if not isinstance(queries, (list, tuple)):
            queries = (queries,) * self.num_sources
        if not isinstance(contexts, (list, tuple)):
            contexts = (contexts,) * self.num_sources
        if len(queries) != self.num_sources:
            raise ValueError(f"expected {self.num_sources} queries, got {len(queries)}")
        if len(contexts) != self.num_sources:
            raise ValueError(f"expected {self.num_sources} contexts, got {len(contexts)}")

        B, N_q, C = queries[0].shape
        outs = []
        attns = []
        for i, (q_tok, context) in enumerate(zip(queries, contexts)):
            q = (
                self.q_proj[i](q_tok)
                .reshape(B, N_q, self.heads_per_source, self.head_dim)
                .permute(0, 2, 1, 3)
            )
            q = self.q_norm(q)
            if self.rope is not None:
                qpos_i = qpos[i] if isinstance(qpos, (list, tuple)) else qpos
                q = self.rope(q, qpos_i)
            N_c = context.shape[1]
            kv = (
                self.kv_proj[i](context)
                .reshape(B, N_c, 2, self.heads_per_source, self.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            k, v = kv.unbind(0)
            k = self.k_norm(k)
            if self.rope is not None:
                kpos_i = kpos[i] if isinstance(kpos, (list, tuple)) else kpos
                k = self.rope(k, kpos_i)
            x_i, attn_i = self._attend(q, k, v, return_attention)
            outs.append(x_i)
            attns.append(attn_i)

        x = torch.cat(outs, dim=1).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj_drop(self.proj(x))
        if not return_attention:
            return x
        attn_maps = [a.mean(dim=1) for a in attns]
        if all(a.shape == attn_maps[0].shape for a in attn_maps[1:]):
            return x, torch.stack(attn_maps, dim=0).mean(dim=0)
        return x, attn_maps
