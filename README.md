# attentions

Goal: learn **basic CUDA** (SIMT, shared memory, tiling) by reimplementing attention kernels next to a small nanoGPT-style model.

GPT layout from [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT).

## Methods

| impl | what | notes |
|------|------|-------|
| `naive` | full `(T,T)` softmax | [docs/naive.md](docs/naive.md) |
| `sdpa` | `F.scaled_dot_product_attention` | [docs/sdpa.md](docs/sdpa.md) |
| `triton` | Triton FA (SRAM tiles) | [docs/triton.md](docs/triton.md) |
| `cuda` | CUDA C++ + `__shared__` | [docs/cuda.md](docs/cuda.md) |
| `numba` | same SIMT idea in Numba | [docs/numba.md](docs/numba.md) |

`triton` / `cuda` / `numba` are forward-only here.

## Bench (my machine)

**GPU:** NVIDIA GeForce RTX 5070 (sm_120) · PyTorch 2.7+cu128 · `tiny`, B=2, T=128, D=32, fp32

| impl | mean ms |
|------|--------:|
| sdpa | 0.037 |
| triton | 0.039 |
| naive | 0.065 |
| cuda | 0.107 |
| numba | 0.364 |
