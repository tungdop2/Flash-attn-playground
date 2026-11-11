from __future__ import annotations

import math

import numba
import torch
from numba import cuda

_MAX_BLOCK_N = 32
_MAX_D = 128
_BLOCK_M = 32


@cuda.jit
def _flash_attn_fwd_kernel(Q, K, V, Out, scale, T, D, block_n):
    bh = cuda.blockIdx.x
    tile_m = cuda.blockIdx.y
    tid = cuda.threadIdx.x
    q_row = tile_m * _BLOCK_M + tid
    active = q_row < T

    sK = cuda.shared.array((_MAX_BLOCK_N, _MAX_D), dtype=numba.float32)
    sV = cuda.shared.array((_MAX_BLOCK_N, _MAX_D), dtype=numba.float32)

    q_reg = cuda.local.array(_MAX_D, dtype=numba.float32)
    acc = cuda.local.array(_MAX_D, dtype=numba.float32)
    scores = cuda.local.array(_MAX_BLOCK_N, dtype=numba.float32)

    m_i = numba.float32(-1.0e30)
    l_i = numba.float32(0.0)

    if active:
        for d in range(D):
            q_reg[d] = Q[bh, q_row, d]
            acc[d] = numba.float32(0.0)

    start = 0
    while start < T:
        cols = block_n if (T - start) > block_n else (T - start)

        idx = tid
        while idx < cols * D:
            row = idx // D
            col = idx % D
            sK[row, col] = K[bh, start + row, col]
            sV[row, col] = V[bh, start + row, col]
            idx += _BLOCK_M
        cuda.syncthreads()

        if active:
            tile_max = numba.float32(-1.0e30)
            for j in range(cols):
                s = numba.float32(0.0)
                for d in range(D):
                    s += q_reg[d] * sK[j, d]
                s = s * scale
                scores[j] = s
                if s > tile_max:
                    tile_max = s

            m_new = m_i if m_i > tile_max else tile_max
            alpha = math.exp(m_i - m_new)

            for d in range(D):
                acc[d] = acc[d] * alpha

            sum_p = numba.float32(0.0)
            for j in range(cols):
                p = math.exp(scores[j] - m_new)
                sum_p += p
                for d in range(D):
                    acc[d] = acc[d] + p * sV[j, d]

            l_i = alpha * l_i + sum_p
            m_i = m_new

        cuda.syncthreads()
        start += block_n

    if active:
        inv_l = numba.float32(1.0) / l_i
        for d in range(D):
            Out[bh, q_row, d] = acc[d] * inv_l


def numba_flash_attention(q, k, v, scale=None, block_size=32):
    assert q.is_cuda and k.is_cuda and v.is_cuda, "numba attn requires CUDA tensors"
    assert q.shape == k.shape == v.shape
    assert q.dim() == 4, f"expected (B, H, T, D), got {q.shape}"
    B, H, T, D = q.shape
    assert 0 < D <= _MAX_D, f"head_dim must be in 1..{_MAX_D}, got {D}"
    assert 0 < block_size <= _MAX_BLOCK_N, (
        f"block_size must be in 1..{_MAX_BLOCK_N}, got {block_size}"
    )
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    in_dtype = q.dtype
    qf = q.contiguous().float().reshape(B * H, T, D)
    kf = k.contiguous().float().reshape(B * H, T, D)
    vf = v.contiguous().float().reshape(B * H, T, D)
    out = torch.empty_like(qf)

    q_nb = cuda.as_cuda_array(qf)
    k_nb = cuda.as_cuda_array(kf)
    v_nb = cuda.as_cuda_array(vf)
    o_nb = cuda.as_cuda_array(out)

    tiles_m = (T + _BLOCK_M - 1) // _BLOCK_M
    _flash_attn_fwd_kernel[(B * H, tiles_m), _BLOCK_M](
        q_nb,
        k_nb,
        v_nb,
        o_nb,
        numba.float32(scale),
        T,
        D,
        int(block_size),
    )
    torch.cuda.synchronize()
    return out.view(B, H, T, D).to(dtype=in_dtype)
