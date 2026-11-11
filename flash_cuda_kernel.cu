#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kMaxD = 128;
constexpr int kMaxBlockN = 64;
constexpr int kBlockM = 32;

__global__ void flash_attn_fwd_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ Out,
    int T,
    int D,
    int block_n,
    float scale) {
  const int bh = blockIdx.x;
  const int tile_m = blockIdx.y;
  const int tid = threadIdx.x;
  const int q_row = tile_m * kBlockM + tid;
  const bool active = q_row < T;

  extern __shared__ float smem[];
  float* sK = smem;
  float* sV = sK + block_n * D;

  float q_reg[kMaxD];
  float acc[kMaxD];
  float m_i = -INFINITY;
  float l_i = 0.f;

  if (active) {
    const float* q_ptr =
        Q + (static_cast<int64_t>(bh) * T + q_row) * D;
#pragma unroll 4
    for (int d = 0; d < D; ++d) {
      q_reg[d] = q_ptr[d];
      acc[d] = 0.f;
    }
  }

  for (int start = 0; start < T; start += block_n) {
    const int cols = block_n < (T - start) ? block_n : (T - start);

    for (int idx = tid; idx < cols * D; idx += kBlockM) {
      const int row = idx / D;
      const int col = idx % D;
      const int64_t global =
          (static_cast<int64_t>(bh) * T + start + row) * D + col;
      sK[idx] = K[global];
      sV[idx] = V[global];
    }
    __syncthreads();

    if (active) {
      float scores[kMaxBlockN];
      float tile_max = -INFINITY;

#pragma unroll 1
      for (int j = 0; j < cols; ++j) {
        float s = 0.f;
#pragma unroll 4
        for (int d = 0; d < D; ++d) {
          s += q_reg[d] * sK[j * D + d];
        }
        s *= scale;
        scores[j] = s;
        tile_max = fmaxf(tile_max, s);
      }

      const float m_new = fmaxf(m_i, tile_max);
      const float alpha = expf(m_i - m_new);

      float sum_p = 0.f;
#pragma unroll 4
      for (int d = 0; d < D; ++d) {
        acc[d] *= alpha;
      }

#pragma unroll 1
      for (int j = 0; j < cols; ++j) {
        const float p = expf(scores[j] - m_new);
        sum_p += p;
#pragma unroll 4
        for (int d = 0; d < D; ++d) {
          acc[d] += p * sV[j * D + d];
        }
      }

      l_i = alpha * l_i + sum_p;
      m_i = m_new;
    }
    __syncthreads();
  }

  if (active) {
    float* out_ptr = Out + (static_cast<int64_t>(bh) * T + q_row) * D;
    const float inv_l = 1.f / l_i;
#pragma unroll 4
    for (int d = 0; d < D; ++d) {
      out_ptr[d] = acc[d] * inv_l;
    }
  }
}

}  // namespace

extern "C" int flash_attn_cuda_launch(
    const float* Q,
    const float* K,
    const float* V,
    float* Out,
    int B,
    int H,
    int T,
    int D,
    int block_n,
    float scale) {
  if (D <= 0 || D > kMaxD || block_n <= 0 || block_n > kMaxBlockN) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const int tiles_m = (T + kBlockM - 1) / kBlockM;
  const dim3 grid(B * H, tiles_m, 1);
  const dim3 block(kBlockM, 1, 1);
  const size_t smem_bytes =
      sizeof(float) * (static_cast<size_t>(block_n) * D * 2);

  flash_attn_fwd_kernel<<<grid, block, smem_bytes>>>(
      Q, K, V, Out, T, D, block_n, scale);

  return static_cast<int>(cudaGetLastError());
}
