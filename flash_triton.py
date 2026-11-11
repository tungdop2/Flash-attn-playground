from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


@triton.jit
def _flash_attn_fwd_kernel(
    Q,
    K,
    V,
    Out,
    sm_scale,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = (
        Q
        + off_z * stride_qz
        + off_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qk
    )
    k_ptrs = (
        K
        + off_z * stride_kz
        + off_h * stride_kh
        + offs_n[None, :] * stride_kn
        + offs_d[:, None] * stride_kk
    )
    v_ptrs = (
        V
        + off_z * stride_vz
        + off_h * stride_vh
        + offs_n[:, None] * stride_vn
        + offs_d[None, :] * stride_vk
    )
    o_ptrs = (
        Out
        + off_z * stride_oz
        + off_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_ok
    )

    mask_m = offs_m < N_CTX

    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504
    q = (q * qk_scale).to(q.dtype)

    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        cols = start_n + offs_n
        mask_n = cols < N_CTX

        k = tl.load(
            k_ptrs + start_n * stride_kn,
            mask=mask_n[None, :],
            other=0.0,
        )
        v = tl.load(
            v_ptrs + start_n * stride_vn,
            mask=mask_n[:, None],
            other=0.0,
        )

        qk = tl.dot(q, k)
        qk = tl.where(mask_n[None, :], qk, float("-inf"))

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])
        l_ij = tl.sum(p, axis=1)
        l_new = alpha * l_i + l_ij

        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
        l_i = l_new

    acc = tl.where(mask_m[:, None], acc / l_i[:, None], 0.0)
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None])


def triton_flash_attention(q, k, v, scale=None, block_m=64, block_n=64):
    assert q.is_cuda and k.is_cuda and v.is_cuda, "triton flash requires CUDA tensors"
    assert q.shape == k.shape == v.shape, (
        f"q/k/v shapes must match, got {q.shape}, {k.shape}, {v.shape}"
    )
    assert q.dim() == 4, f"expected (B, H, T, D), got {q.shape}"
    assert q.dtype in (torch.float16, torch.bfloat16, torch.float32), q.dtype
    assert block_m > 0 and block_n > 0

    B, H, T, D = q.shape
    assert _is_power_of_2(D), f"head_dim must be power of 2 for this kernel, got {D}"
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = torch.empty_like(q)

    grid = (triton.cdiv(T, block_m), B * H)
    _flash_attn_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        float(scale),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        B,
        H,
        T,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_DMODEL=D,
    )
    return out
