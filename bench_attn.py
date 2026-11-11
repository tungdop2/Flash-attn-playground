from __future__ import annotations

import argparse
import math
import statistics
import time

import torch

from attention import ATTN_IMPLS, run_attention
from configs import get_config, list_configs
from model import GPT


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _device_from_str(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def benchmark_fn(fn, *, warmup: int, iters: int, device: torch.device) -> list[float]:
    for _ in range(warmup):
        fn()
    _sync(device)

    times_ms = []
    for _ in range(iters):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return times_ms


def make_qkv(B, H, T, D, device, dtype):
    q = torch.randn(B, H, T, D, device=device, dtype=dtype)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype)
    return q, k, v


def bench_kernels(args) -> None:
    device = _device_from_str(args.device)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        args.dtype
    ]
    if device.type == "cpu" and dtype != torch.float32:
        print(f"warning: {args.dtype} on CPU may be slow/unsupported; using fp32")
        dtype = torch.float32

    cfg = get_config(args.config)
    B = args.batch
    H = cfg.n_heads
    D = cfg.n_embed // cfg.n_heads
    scale = 1.0 / math.sqrt(D)

    print(
        f"[kernel] device={device} dtype={dtype} "
        f"B={B} H={H} T={args.seq} D={D} block={args.block_size}"
    )
    print(f"{'impl':<8} {'mean_ms':>10} {'std_ms':>10} {'min_ms':>10} {'max_ms':>10}")

    for impl in args.impls:
        q, k, v = make_qkv(B, H, args.seq, D, device, dtype)

        def run(impl=impl, q=q, k=k, v=v):
            return run_attention(
                impl,
                q,
                k,
                v,
                scale,
                dropout_p=0.0,
                training=False,
                block_size=args.block_size,
            )

        if impl != "naive" and args.seq <= 256:
            with torch.no_grad():
                ref = run_attention(
                    "naive", q.float(), k.float(), v.float(), scale, training=False
                )
                out = run().float()
                diff = (ref - out).abs().max().item()
                tol = (
                    5e-3
                    if impl in ("triton", "cuda", "numba") or dtype != torch.float32
                    else 1e-3
                )
                assert diff < tol, f"{impl} mismatch vs naive: {diff}"

        times = benchmark_fn(
            lambda: run(),
            warmup=args.warmup,
            iters=args.iters,
            device=device,
        )
        print(
            f"{impl:<8} {statistics.mean(times):10.3f} "
            f"{statistics.pstdev(times):10.3f} "
            f"{min(times):10.3f} {max(times):10.3f}"
        )


def bench_model(args) -> None:
    device = _device_from_str(args.device)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        args.dtype
    ]
    if device.type == "cpu" and dtype != torch.float32:
        print(f"warning: {args.dtype} on CPU may be slow/unsupported; using fp32")
        dtype = torch.float32

    base = get_config(args.config, dropout=0.0, flash_block_size=args.block_size)
    assert args.seq <= base.max_length, (
        f"seq {args.seq} > max_length {base.max_length}; pick a larger config"
    )

    print(
        f"[model] device={device} dtype={dtype} "
        f"config={args.config} B={args.batch} T={args.seq}"
    )
    print(f"{'impl':<8} {'mean_ms':>10} {'std_ms':>10} {'min_ms':>10} {'max_ms':>10}")

    ref_state = None
    for impl in args.impls:
        cfg = base.with_updates(attn_impl=impl)
        model = GPT(cfg, verbose=False).to(device=device, dtype=dtype).eval()
        if ref_state is None:
            ref_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            model.load_state_dict(
                {k: v.to(device=device, dtype=dtype) for k, v in ref_state.items()}
            )

        idx = torch.randint(0, cfg.vocab_size, (args.batch, args.seq), device=device)

        def run(model=model, idx=idx):
            with torch.no_grad():
                return model(idx)

        times = benchmark_fn(
            run,
            warmup=args.warmup,
            iters=args.iters,
            device=device,
        )
        print(
            f"{impl:<8} {statistics.mean(times):10.3f} "
            f"{statistics.pstdev(times):10.3f} "
            f"{min(times):10.3f} {max(times):10.3f}"
        )


def parse_args():
    p = argparse.ArgumentParser(description="Compare attention implementation latency")
    p.add_argument("--mode", choices=("kernel", "model", "both"), default="both")
    p.add_argument("--config", default="bench", choices=list_configs())
    p.add_argument("--impls", nargs="+", default=list(ATTN_IMPLS), choices=list(ATTN_IMPLS))
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0 ...")
    p.add_argument("--dtype", default="fp32", choices=("fp32", "fp16", "bf16"))
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode in ("kernel", "both"):
        bench_kernels(args)
        print()
    if args.mode in ("model", "both"):
        bench_model(args)


if __name__ == "__main__":
    main()
