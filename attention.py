import math

import torch
from torch import nn
from torch.nn import functional as F

ATTN_IMPLS = ("naive", "sdpa", "triton", "cuda", "numba")


def naive_attention(q, k, v, scale, dropout_p=0.0, training=True):
    assert q.shape == k.shape == v.shape, (
        f"q/k/v shapes must match, got {q.shape}, {k.shape}, {v.shape}"
    )
    assert q.dim() == 4, f"expected (B, H, T, D), got {q.shape}"
    att = (q @ k.transpose(-2, -1)) * scale
    att = F.softmax(att, dim=-1)
    if dropout_p > 0.0 and training:
        att = F.dropout(att, p=dropout_p)
    return att @ v


def sdpa_attention(q, k, v, scale, dropout_p=0.0, training=True, is_causal=False):
    assert q.shape == k.shape == v.shape, (
        f"q/k/v shapes must match, got {q.shape}, {k.shape}, {v.shape}"
    )
    assert q.dim() == 4, f"expected (B, H, T, D), got {q.shape}"
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=dropout_p if training else 0.0,
        is_causal=is_causal,
        scale=scale,
    )


def triton_attention(q, k, v, scale, block_size=64):
    from flash_triton import triton_flash_attention

    assert q.is_cuda, "attn_impl='triton' requires CUDA tensors"
    return triton_flash_attention(
        q, k, v, scale=scale, block_m=block_size, block_n=block_size
    )


def cuda_attention(q, k, v, scale, block_size=32):
    from flash_cuda import cuda_flash_attention

    assert q.is_cuda, "attn_impl='cuda' requires CUDA tensors"
    return cuda_flash_attention(q, k, v, scale=scale, block_size=block_size)


def numba_attention(q, k, v, scale, block_size=32):
    from flash_numba import numba_flash_attention

    assert q.is_cuda, "attn_impl='numba' requires CUDA tensors"
    return numba_flash_attention(q, k, v, scale=scale, block_size=block_size)


def run_attention(attn_impl, q, k, v, scale, *, dropout_p=0.0, training=False, block_size=64):
    assert attn_impl in ATTN_IMPLS, f"unknown attn_impl={attn_impl!r}"
    if attn_impl == "naive":
        return naive_attention(q, k, v, scale, dropout_p=dropout_p, training=training)
    if attn_impl == "triton":
        assert not (training and dropout_p > 0.0), (
            "triton attn_impl does not support attention dropout; set dropout=0"
        )
        assert not training, (
            "triton attn_impl is forward-only (no backward); use eval() or sdpa/naive"
        )
        return triton_attention(q, k, v, scale, block_size=block_size)
    if attn_impl == "cuda":
        assert not (training and dropout_p > 0.0), (
            "cuda attn_impl does not support attention dropout; set dropout=0"
        )
        assert not training, (
            "cuda attn_impl is forward-only (no backward); use eval() or sdpa/naive"
        )
        return cuda_attention(q, k, v, scale, block_size=min(block_size, 64))
    if attn_impl == "numba":
        assert not (training and dropout_p > 0.0), (
            "numba attn_impl does not support attention dropout; set dropout=0"
        )
        assert not training, (
            "numba attn_impl is forward-only (no backward); use eval() or sdpa/naive"
        )
        return numba_attention(q, k, v, scale, block_size=min(block_size, 32))
    return sdpa_attention(q, k, v, scale, dropout_p=dropout_p, training=training)


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_heads == 0, (
            f"n_embed ({config.n_embed}) must be divisible by n_heads ({config.n_heads})"
        )
        assert config.attn_impl in ATTN_IMPLS, (
            f"attn_impl must be one of {ATTN_IMPLS}, got {config.attn_impl!r}"
        )
        assert config.flash_block_size > 0, (
            f"flash_block_size must be positive, got {config.flash_block_size}"
        )

        self.config = config
        self.n_embed = config.n_embed
        self.n_heads = config.n_heads
        self.head_dim = self.n_embed // self.n_heads
        self.attn_impl = config.attn_impl
        self.flash_block_size = config.flash_block_size

        self.c_attn = nn.Linear(self.n_embed, 3 * self.n_embed)
        self.c_proj = nn.Linear(self.n_embed, self.n_embed)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()
        assert C == self.n_embed, f"expected C={self.n_embed}, got {C}"
        assert T <= self.config.max_length, (
            f"sequence length {T} exceeds max_length {self.config.max_length}"
        )

        qkv = self.c_attn(x)
        assert qkv.shape == (B, T, 3 * self.n_embed)
        q, k, v = qkv.split(self.n_embed, dim=2)

        q = q.view(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        assert q.shape == (B, self.n_heads, T, self.head_dim)

        scale = 1.0 / math.sqrt(self.head_dim)
        y = run_attention(
            self.attn_impl,
            q,
            k,
            v,
            scale,
            dropout_p=self.attn_dropout.p,
            training=self.training,
            block_size=self.flash_block_size,
        )

        assert y.shape == (B, self.n_heads, T, self.head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        assert y.shape == (B, T, C)
        return y
