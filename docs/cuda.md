# `cuda`

Goal here: basic CUDA C++ — grid/block, `__shared__`, `__syncthreads__`, then a FlashAttention-style forward.

Idea: K/V tiles in shared mem, one thread per query in a Q tile, online softmax.

Refs:

- [Intro to CUDA C++](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html)
- [Writing CUDA Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
