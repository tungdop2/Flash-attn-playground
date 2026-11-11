from __future__ import annotations

import ctypes
import math
import os
import subprocess
import sys
from pathlib import Path

import torch

_DIR = Path(__file__).resolve().parent
_SRC = _DIR / "flash_cuda_kernel.cu"
_SO = _DIR / "flash_cuda_kernel.so"
_LIB = None


def _nvcc() -> str:
    for cand in (
        os.environ.get("CUDACXX"),
        os.environ.get("NVCC"),
        "/usr/local/cuda/bin/nvcc",
        "nvcc",
    ):
        if not cand:
            continue
        path = Path(cand)
        if path.is_file() or cand == "nvcc":
            return cand
    raise RuntimeError("nvcc not found; install CUDA toolkit")


def _build_so(force: bool = False) -> Path:
    if _SO.exists() and not force:
        if _SO.stat().st_mtime >= _SRC.stat().st_mtime:
            return _SO

    arch = os.environ.get("CUDA_ARCH", "90")
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        arch = f"{major}{minor}"

    cmd = [
        _nvcc(),
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-O3",
        "--use_fast_math",
        f"-gencode=arch=compute_{arch},code=sm_{arch}",
        str(_SRC),
        "-o",
        str(_SO),
    ]
    if os.environ.get("FLASH_CUDA_VERBOSE", "0") == "1":
        print("Building CUDA kernel:", " ".join(cmd), file=sys.stderr)
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return _SO


def _get_lib():
    global _LIB
    if _LIB is None:
        so = _build_so()
        _LIB = ctypes.CDLL(str(so))
        _LIB.flash_attn_cuda_launch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        ]
        _LIB.flash_attn_cuda_launch.restype = ctypes.c_int
    return _LIB


def cuda_flash_attention(q, k, v, scale=None, block_size=32):
    assert q.is_cuda and k.is_cuda and v.is_cuda, "cuda attn requires CUDA tensors"
    assert q.shape == k.shape == v.shape
    assert q.dim() == 4, f"expected (B, H, T, D), got {q.shape}"
    B, H, T, D = q.shape
    assert 0 < D <= 128, f"head_dim must be in 1..128, got {D}"
    assert 0 < block_size <= 64, f"block_size must be in 1..64, got {block_size}"
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    in_dtype = q.dtype
    qf = q.contiguous().float()
    kf = k.contiguous().float()
    vf = v.contiguous().float()
    out = torch.empty_like(qf)

    lib = _get_lib()
    err = lib.flash_attn_cuda_launch(
        qf.data_ptr(),
        kf.data_ptr(),
        vf.data_ptr(),
        out.data_ptr(),
        int(B),
        int(H),
        int(T),
        int(D),
        int(block_size),
        float(scale),
    )
    if err != 0:
        raise RuntimeError(f"flash_attn_cuda_launch failed with cudaError={err}")
    torch.cuda.synchronize()
    return out.to(dtype=in_dtype)
