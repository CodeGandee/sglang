/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <cfloat>
#include <cstdint>
#include <limits>

#include "utils.h"

namespace {

constexpr int kHeadDim = 64;
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarps = kThreads / kWarpSize;
constexpr int kReferenceGemmInterleavedThreshold = 1760;
constexpr int kReferenceSoftmaxLargeRowThreshold = 2048;
constexpr int kReferenceSoftmaxThreads = 1024;

void check_b200(const at::Tensor& tensor) {
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, tensor.get_device()));
  TORCH_CHECK(
      properties.major == 10 && properties.minor == 0,
      "shadowkv_packed_gqa requires NVIDIA B200 compute capability 10.0; found ",
      properties.major,
      ".",
      properties.minor);
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  return value;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ float block_max(float value, float* shared) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  value = warp_max(value);
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();
  value = threadIdx.x < kWarps ? shared[lane] : -FLT_MAX;
  if (warp == 0) {
    value = warp_max(value);
  }
  if (threadIdx.x == 0) {
    shared[0] = value;
  }
  __syncthreads();
  return shared[0];
}

__device__ __forceinline__ float block_sum(float value, float* shared) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  value = warp_sum(value);
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();
  value = threadIdx.x < kWarps ? shared[lane] : 0.0f;
  if (warp == 0) {
    value = warp_sum(value);
  }
  if (threadIdx.x == 0) {
    shared[0] = value;
  }
  __syncthreads();
  return shared[0];
}

__global__ void shadowkv_packed_gqa_kernel(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ keys,
    const __nv_bfloat16* __restrict__ values,
    const int32_t* __restrict__ lengths,
    float* __restrict__ weights,
    __nv_bfloat16* __restrict__ output,
    int batch_size,
    int query_heads,
    int kv_heads,
    int maximum_tokens) {
  const int query_row = blockIdx.x;
  const int batch = query_row / query_heads;
  const int query_head = query_row - batch * query_heads;
  if (batch >= batch_size) {
    return;
  }
  const int groups = query_heads / kv_heads;
  const int kv_head = query_head / groups;
  const int length = max(0, min(lengths[batch], maximum_tokens));
  const int64_t query_offset =
      (static_cast<int64_t>(batch) * query_heads + query_head) * kHeadDim;
  const int64_t kv_base =
      (static_cast<int64_t>(batch) * kv_heads + kv_head) * maximum_tokens * kHeadDim;
  const int64_t weight_base =
      (static_cast<int64_t>(batch) * query_heads + query_head) * maximum_tokens;

  if (length == 0) {
    if (threadIdx.x < kHeadDim) {
      output[query_offset + threadIdx.x] = __float2bfloat16_rn(0.0f);
    }
    return;
  }

  __shared__ float reduction[kReferenceSoftmaxThreads];
  float local_max = -FLT_MAX;
  for (int token = threadIdx.x; token < length; token += blockDim.x) {
    const int64_t key_offset = kv_base + static_cast<int64_t>(token) * kHeadDim;
    float score = 0.0f;
    // The locked B200 PyTorch GEMM switches to sixteen interleaved K-lane
    // accumulators above this qualified Llama GQA row width. Matching that
    // order prevents small score errors from changing greedy continuations.
    if (length > kReferenceGemmInterleavedThreshold) {
      float partial[16];
#pragma unroll
      for (int lane = 0; lane < 16; ++lane) {
        float accumulated = 0.0f;
#pragma unroll
        for (int dimension = lane; dimension < kHeadDim; dimension += 16) {
          accumulated += __bfloat162float(query[query_offset + dimension]) *
              __bfloat162float(keys[key_offset + dimension]);
        }
        partial[lane] = accumulated;
      }
      score = partial[0];
#pragma unroll
      for (int lane = 1; lane < 16; ++lane) {
        score += partial[lane];
      }
    } else {
#pragma unroll
      for (int dimension = 0; dimension < kHeadDim; ++dimension) {
        score += __bfloat162float(query[query_offset + dimension]) *
            __bfloat162float(keys[key_offset + dimension]);
      }
    }
    score *= 0.125f;
    weights[weight_base + token] = score;
    local_max = fmaxf(local_max, score);
  }
  const float maximum = block_max(local_max, reduction);

  float denominator;
  if (length > kReferenceSoftmaxLargeRowThreshold) {
    // PyTorch uses 1,024 logical lanes for large FP32 softmax rows. Populate
    // those lanes with 256 physical threads, then preserve its two-level
    // warp reduction order exactly.
    for (int lane = threadIdx.x; lane < kReferenceSoftmaxThreads;
         lane += blockDim.x) {
      float local_sum = 0.0f;
      for (int token = lane; token < length;
           token += kReferenceSoftmaxThreads) {
        local_sum += expf(weights[weight_base + token] - maximum);
      }
      reduction[lane] = local_sum;
    }
    __syncthreads();
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
      for (int lane = threadIdx.x; lane < kReferenceSoftmaxThreads;
           lane += blockDim.x) {
        if (lane % kWarpSize < offset) {
          reduction[lane] += reduction[lane + offset];
        }
      }
      __syncthreads();
    }
    float warp_total =
        threadIdx.x < kWarpSize ? reduction[threadIdx.x * kWarpSize] : 0.0f;
    if (threadIdx.x < kWarpSize) {
      warp_total = warp_sum(warp_total);
    }
    if (threadIdx.x == 0) {
      reduction[0] = warp_total;
    }
    __syncthreads();
    denominator = reduction[0];
  } else {
    float local_sum = 0.0f;
    for (int token = threadIdx.x; token < length; token += blockDim.x) {
      local_sum += expf(weights[weight_base + token] - maximum);
    }
    denominator = block_sum(local_sum, reduction);
  }
  for (int token = threadIdx.x; token < length; token += blockDim.x) {
    weights[weight_base + token] =
        expf(weights[weight_base + token] - maximum) / denominator;
  }
  __syncthreads();

  if (threadIdx.x < kHeadDim) {
    const int dimension = threadIdx.x;
    float result;
    if (length > kReferenceSoftmaxLargeRowThreshold) {
      // FP64 accumulation stabilizes the final BF16 rounding at long context
      // without changing the FP32 softmax contract above.
      double accumulated = 0.0;
      for (int token = 0; token < length; ++token) {
        const int64_t value_offset =
            kv_base + static_cast<int64_t>(token) * kHeadDim + dimension;
        accumulated += static_cast<double>(weights[weight_base + token]) *
            static_cast<double>(__bfloat162float(values[value_offset]));
      }
      result = static_cast<float>(accumulated);
    } else {
      float accumulated = 0.0f;
      for (int token = 0; token < length; ++token) {
        const int64_t value_offset =
            kv_base + static_cast<int64_t>(token) * kHeadDim + dimension;
        accumulated += weights[weight_base + token] *
            __bfloat162float(values[value_offset]);
      }
      result = accumulated;
    }
    output[query_offset + dimension] = __float2bfloat16_rn(result);
  }
}

}  // namespace

void shadowkv_packed_gqa(
    const at::Tensor& query,
    const at::Tensor& keys,
    const at::Tensor& values,
    const at::Tensor& lengths,
    at::Tensor& weights,
    at::Tensor& output) {
  CHECK_INPUT(query);
  CHECK_INPUT(keys);
  CHECK_INPUT(values);
  CHECK_INPUT(lengths);
  CHECK_INPUT(weights);
  CHECK_INPUT(output);
  TORCH_CHECK(query.scalar_type() == at::ScalarType::BFloat16, "query must use bfloat16");
  TORCH_CHECK(keys.scalar_type() == at::ScalarType::BFloat16, "keys must use bfloat16");
  TORCH_CHECK(values.scalar_type() == at::ScalarType::BFloat16, "values must use bfloat16");
  TORCH_CHECK(output.scalar_type() == at::ScalarType::BFloat16, "output must use bfloat16");
  TORCH_CHECK(lengths.scalar_type() == at::ScalarType::Int, "lengths must use int32");
  TORCH_CHECK(weights.scalar_type() == at::ScalarType::Float, "weights must use float32");
  TORCH_CHECK(query.dim() == 3 && query.size(2) == kHeadDim, "query must have shape [batch, q_heads, 64]");
  TORCH_CHECK(keys.dim() == 4 && keys.size(3) == kHeadDim, "keys must have shape [batch, kv_heads, tokens, 64]");
  TORCH_CHECK(values.sizes() == keys.sizes(), "values must match the packed key shape");
  TORCH_CHECK(query.size(0) == keys.size(0), "query and packed KV batch dimensions differ");
  TORCH_CHECK(query.size(1) % keys.size(1) == 0, "query heads must be divisible by KV heads");
  TORCH_CHECK(lengths.dim() == 1 && lengths.size(0) == query.size(0), "lengths must have shape [batch]");
  TORCH_CHECK(
      weights.dim() == 3 && weights.size(0) == query.size(0) && weights.size(1) == query.size(1) &&
          weights.size(2) == keys.size(2),
      "weights must have shape [batch, q_heads, tokens]");
  TORCH_CHECK(output.sizes() == query.sizes(), "output must match the query shape");
  TORCH_CHECK(
      query.device() == keys.device() && query.device() == values.device() && query.device() == lengths.device() &&
          query.device() == weights.device() && query.device() == output.device(),
      "all shadowkv_packed_gqa tensors must share one CUDA device");
  TORCH_CHECK(query.size(0) > 0 && query.size(1) > 0 && keys.size(1) > 0, "packed GQA dimensions must be positive");
  TORCH_CHECK(keys.size(2) > 0, "packed GQA token capacity must be positive");
  TORCH_CHECK(
      query.size(0) * query.size(1) <= static_cast<int64_t>(std::numeric_limits<int>::max()),
      "packed GQA grid exceeds the CUDA launch bound");

  c10::cuda::CUDAGuard device_guard(query.device());
  check_b200(query);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int blocks = static_cast<int>(query.size(0) * query.size(1));
  shadowkv_packed_gqa_kernel<<<blocks, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(query.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(keys.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(values.data_ptr<at::BFloat16>()),
      lengths.data_ptr<int32_t>(),
      weights.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
      static_cast<int>(query.size(0)),
      static_cast<int>(query.size(1)),
      static_cast<int>(keys.size(1)),
      static_cast<int>(keys.size(2)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
