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

#include <limits>

#include "shadowkv/device_contract.cuh"
#include "utils.h"

namespace {

constexpr int kHeadDim = 64;
constexpr int kHalfHeadDim = kHeadDim / 2;
constexpr int kRank = 160;
constexpr int kGeneralHeadDim = 128;
constexpr int kGatherThreads = 256;

bool supported_general_rank(int64_t rank) {
  return rank == 64 || rank == 128 || rank == 160 || rank == 256;
}

__global__ void shadowkv_reconstruct_rope_kernel(
    const __nv_bfloat16* __restrict__ reconstructed,
    const float* __restrict__ cosine,
    const float* __restrict__ sine,
    __nv_bfloat16* __restrict__ output,
    int selected_tokens,
    int64_t output_head_stride,
    int64_t output_token_stride) {
  const int dimension = threadIdx.x;
  const int head = blockIdx.x / selected_tokens;
  const int selected = blockIdx.x - head * selected_tokens;

  const int64_t reconstructed_offset =
      (static_cast<int64_t>(head) * selected_tokens + selected) * kHeadDim;
  const int64_t output_offset =
      static_cast<int64_t>(head) * output_head_stride +
      static_cast<int64_t>(selected) * output_token_stride;
  const float value =
      __bfloat162float(reconstructed[reconstructed_offset + dimension]);

  const int frequency_index = dimension % kHalfHeadDim;
  const int64_t frequency_offset =
      (static_cast<int64_t>(head) * selected_tokens + selected) * kHalfHeadDim +
      frequency_index;
  const int paired_dimension =
      dimension < kHalfHeadDim ? dimension + kHalfHeadDim : dimension - kHalfHeadDim;
  const float paired_value =
      __bfloat162float(reconstructed[reconstructed_offset + paired_dimension]);
  const float rotated_half = dimension < kHalfHeadDim ? -paired_value : paired_value;
  // Match the readable Torch expression's two rounded FP32 products followed
  // by a rounded FP32 addition. Explicit intrinsics prevent FMA contraction.
  const float direct_product = __fmul_rn(value, cosine[frequency_offset]);
  const float rotated_product = __fmul_rn(rotated_half, sine[frequency_offset]);
  output[output_offset + dimension] =
      __float2bfloat16_rn(__fadd_rn(direct_product, rotated_product));
}

template <int Rank>
__global__ void shadowkv_gather_u_kernel(
    const __nv_bfloat16* __restrict__ u,
    const int64_t* __restrict__ positions,
    __nv_bfloat16* __restrict__ gathered_u,
    int selected_tokens) {
  const int head = blockIdx.x / selected_tokens;
  const int selected = blockIdx.x - head * selected_tokens;
  const int64_t source_row =
      positions[static_cast<int64_t>(head) * selected_tokens + selected];
  const int64_t source_offset = source_row * Rank;
  const int64_t destination_offset =
      (static_cast<int64_t>(head) * selected_tokens + selected) * Rank;
  for (int column = threadIdx.x; column < Rank; column += blockDim.x) {
    gathered_u[destination_offset + column] = u[source_offset + column];
  }
}

template <int Rank>
void launch_shadowkv_gather_u(
    const at::Tensor& u,
    const at::Tensor& positions,
    at::Tensor& gathered_u,
    int blocks,
    cudaStream_t stream) {
  shadowkv_gather_u_kernel<Rank><<<blocks, kGatherThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(u.data_ptr<at::BFloat16>()),
      positions.data_ptr<int64_t>(),
      reinterpret_cast<__nv_bfloat16*>(gathered_u.data_ptr<at::BFloat16>()),
      static_cast<int>(positions.size(1)));
}

}  // namespace

void shadowkv_reconstruct(
    const at::Tensor& u,
    const at::Tensor& sv,
    const at::Tensor& positions,
    at::Tensor& output) {
  CHECK_INPUT(u);
  CHECK_INPUT(sv);
  CHECK_INPUT(positions);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(output);
  TORCH_CHECK(u.scalar_type() == at::ScalarType::BFloat16, "u must use bfloat16");
  TORCH_CHECK(sv.scalar_type() == at::ScalarType::BFloat16, "sv must use bfloat16");
  TORCH_CHECK(
      output.scalar_type() == at::ScalarType::BFloat16,
      "output must use bfloat16");
  TORCH_CHECK(
      positions.scalar_type() == at::ScalarType::Long,
      "positions must use int64");
  TORCH_CHECK(u.dim() == 2, "u must have shape [tokens, rank]");
  TORCH_CHECK(
      supported_general_rank(u.size(1)),
      "rank must be one of 64, 128, 160, or 256");
  TORCH_CHECK(
      sv.dim() == 3 && sv.size(1) == u.size(1) &&
          sv.size(2) == kGeneralHeadDim,
      "sv must have shape [kv_heads, rank, 128]");
  TORCH_CHECK(
      positions.dim() == 2,
      "positions must have shape [kv_heads, selected_tokens]");
  TORCH_CHECK(
      positions.size(0) == sv.size(0),
      "positions and sv must have the same kv_heads");
  TORCH_CHECK(
      output.dim() == 3 && output.size(0) == sv.size(0) &&
          output.size(1) == positions.size(1) &&
          output.size(2) == kGeneralHeadDim,
      "output must have shape [kv_heads, selected_tokens, 128]");
  TORCH_CHECK(
      u.device() == sv.device() && u.device() == positions.device() &&
          u.device() == output.device(),
      "all shadowkv_reconstruct tensors must share one CUDA device");
  TORCH_CHECK(u.size(0) > 0, "u must contain at least one token");

  c10::cuda::CUDAGuard device_guard(u.device());
  sglang::shadowkv::check_operation_device(
      u,
      "shadowkv_reconstruct");
  if (positions.numel() == 0) {
    return;
  }
  const int64_t minimum_position = positions.min().item<int64_t>();
  const int64_t maximum_position = positions.max().item<int64_t>();
  TORCH_CHECK(minimum_position >= 0, "positions must be nonnegative");
  TORCH_CHECK(maximum_position < u.size(0), "positions exceed the U token dimension");
  TORCH_CHECK(
      sv.size(0) * positions.size(1) <=
          static_cast<int64_t>(std::numeric_limits<int>::max()),
      "reconstruction grid exceeds the CUDA launch bound");

  at::Tensor gathered_u = at::empty(
      {sv.size(0), positions.size(1), u.size(1)}, u.options());
  const int blocks = static_cast<int>(sv.size(0) * positions.size(1));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  switch (u.size(1)) {
    case 64:
      launch_shadowkv_gather_u<64>(u, positions, gathered_u, blocks, stream);
      break;
    case 128:
      launch_shadowkv_gather_u<128>(u, positions, gathered_u, blocks, stream);
      break;
    case 160:
      launch_shadowkv_gather_u<160>(u, positions, gathered_u, blocks, stream);
      break;
    case 256:
      launch_shadowkv_gather_u<256>(u, positions, gathered_u, blocks, stream);
      break;
    default:
      TORCH_CHECK(false, "unsupported reconstruction rank");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const at::Tensor reconstructed =
      torch::einsum("hnr,hrd->hnd", {gathered_u, sv});
  output.copy_(reconstructed);
}

void shadowkv_reconstruct_rope(
    const at::Tensor& u,
    const at::Tensor& sv,
    const at::Tensor& positions,
    const at::Tensor& inverse_frequencies,
    at::Tensor& output) {
  CHECK_INPUT(u);
  CHECK_INPUT(sv);
  CHECK_INPUT(positions);
  CHECK_INPUT(inverse_frequencies);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(output);
  TORCH_CHECK(u.scalar_type() == at::ScalarType::BFloat16, "u must use bfloat16");
  TORCH_CHECK(sv.scalar_type() == at::ScalarType::BFloat16, "sv must use bfloat16");
  TORCH_CHECK(
      output.scalar_type() == at::ScalarType::BFloat16,
      "output must use bfloat16");
  TORCH_CHECK(
      positions.scalar_type() == at::ScalarType::Long,
      "positions must use int64");
  TORCH_CHECK(
      inverse_frequencies.scalar_type() == at::ScalarType::Float,
      "inverse_frequencies must use float32");
  TORCH_CHECK(u.dim() == 2 && u.size(1) == kRank, "u must have shape [tokens, 160]");
  TORCH_CHECK(
      sv.dim() == 3 && sv.size(1) == kRank && sv.size(2) == kHeadDim,
      "sv must have shape [kv_heads, 160, 64]");
  TORCH_CHECK(
      positions.dim() == 2,
      "positions must have shape [kv_heads, selected_tokens]");
  TORCH_CHECK(
      positions.size(0) == sv.size(0),
      "positions and sv must have the same kv_heads");
  TORCH_CHECK(
      inverse_frequencies.dim() == 1 && inverse_frequencies.numel() == kHalfHeadDim,
      "inverse_frequencies must have shape [32]");
  TORCH_CHECK(
      output.dim() == 3 && output.size(0) == sv.size(0) && output.size(1) == positions.size(1) &&
          output.size(2) == kHeadDim,
      "output must have shape [kv_heads, selected_tokens, 64]");
  TORCH_CHECK(
      u.device() == sv.device() && u.device() == positions.device() &&
          u.device() == inverse_frequencies.device() &&
          u.device() == output.device(),
      "all shadowkv_reconstruct_rope tensors must share one CUDA device");
  TORCH_CHECK(u.size(0) > 0, "u must contain at least one token");

  c10::cuda::CUDAGuard device_guard(u.device());
  sglang::shadowkv::check_operation_device(
      u,
      "shadowkv_reconstruct_rope");
  if (positions.numel() == 0) {
    return;
  }
  const int64_t minimum_position = positions.min().item<int64_t>();
  const int64_t maximum_position = positions.max().item<int64_t>();
  TORCH_CHECK(minimum_position >= 0, "positions must be nonnegative");
  TORCH_CHECK(maximum_position < u.size(0), "positions exceed the U token dimension");
  TORCH_CHECK(
      sv.size(0) * positions.size(1) <= static_cast<int64_t>(std::numeric_limits<int>::max()),
      "reconstruction grid exceeds the CUDA launch bound");

  // Use the same BF16 tensor-core matrix product as the readable Torch
  // contract. It accumulates in FP32 and materializes BF16 before the custom
  // RoPE kernel, avoiding reduction-order drift across autoregressive steps.
  const at::Tensor gathered_u = u.index({positions});
  const at::Tensor reconstructed =
      torch::einsum("hnr,hrd->hnd", {gathered_u, sv});
  const at::Tensor angles =
      positions.to(at::ScalarType::Float).unsqueeze(-1) * inverse_frequencies;
  const at::Tensor cosine = angles.cos();
  const at::Tensor sine = angles.sin();
  const int blocks = static_cast<int>(sv.size(0) * positions.size(1));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  shadowkv_reconstruct_rope_kernel<<<blocks, kHeadDim, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(
          reconstructed.data_ptr<at::BFloat16>()),
      cosine.data_ptr<float>(),
      sine.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
      static_cast<int>(positions.size(1)),
      output.stride(0),
      output.stride(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
