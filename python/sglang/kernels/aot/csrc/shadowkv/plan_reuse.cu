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
#include <cuda_runtime.h>
#include <torch/all.h>

#include <limits>

#include "shadowkv/device_contract.cuh"
#include "utils.h"

namespace {

constexpr int64_t kIgnored = 0;
constexpr int64_t kHit = 1;
constexpr int64_t kMiss = 2;
constexpr int32_t kInvalidLength = 1;
constexpr int32_t kInvalidChunk = 2;
constexpr int kPlannerThreads = 256;
constexpr int kMaximumChunkWidth = 256;
constexpr int kMaximumExactWidth = 64;

__global__ void shadowkv_plan_reuse_kernel(
    const int64_t* __restrict__ previous_chunks,
    const int32_t* __restrict__ previous_lengths,
    const int64_t* __restrict__ current_chunks,
    const int32_t* __restrict__ current_lengths,
    const int64_t* __restrict__ exact_chunks,
    const int32_t* __restrict__ exact_lengths,
    const int64_t* __restrict__ cached_generations,
    const int64_t* __restrict__ current_generations,
    int previous_width,
    int current_width,
    int exact_width,
    int max_reuse_chunks,
    int chunk_size,
    int64_t* __restrict__ plan,
    int64_t* __restrict__ deduplicated_exact_chunks,
    int32_t* __restrict__ counts,
    int32_t* __restrict__ error_codes) {
  const int row = blockIdx.x;
  const int thread = threadIdx.x;
  __shared__ int deduplicated_exact_count;

  const int previous_length = previous_lengths[row];
  const int current_length = current_lengths[row];
  const int exact_length = exact_lengths[row];
  if (thread == 0) {
    error_codes[row] = 0;
    counts[row * 3] = 0;
    counts[row * 3 + 1] = 0;
    counts[row * 3 + 2] = 0;
    if (previous_length < 0 || previous_length > previous_width || current_length < 0 ||
        current_length > current_width || exact_length < 0 || exact_length > exact_width) {
      error_codes[row] = kInvalidLength;
    }
    deduplicated_exact_count = 0;
  }
  __syncthreads();
  if (error_codes[row] != 0) {
    return;
  }

  for (int index = thread; index < current_width; index += blockDim.x) {
    const int64_t plan_offset = (static_cast<int64_t>(row) * current_width + index) * 3;
    plan[plan_offset] = -1;
    plan[plan_offset + 1] = -1;
    plan[plan_offset + 2] = -1;
  }
  for (int index = thread; index < exact_width; index += blockDim.x) {
    deduplicated_exact_chunks[row * exact_width + index] = -1;
  }
  __syncthreads();

  for (int index = thread; index < previous_length; index += blockDim.x) {
    if (previous_chunks[row * previous_width + index] < 0) {
      atomicExch(&error_codes[row], kInvalidChunk);
    }
  }
  for (int index = thread; index < current_length; index += blockDim.x) {
    if (current_chunks[row * current_width + index] < 0) {
      atomicExch(&error_codes[row], kInvalidChunk);
    }
  }
  for (int index = thread; index < exact_length; index += blockDim.x) {
    if (exact_chunks[row * exact_width + index] < 0) {
      atomicExch(&error_codes[row], kInvalidChunk);
    }
  }
  __syncthreads();
  if (error_codes[row] != 0) {
    return;
  }

  if (thread == 0) {
    for (int index = 0; index < exact_length; ++index) {
      const int64_t chunk = exact_chunks[row * exact_width + index];
      bool duplicate = false;
      for (int prior = 0; prior < index; ++prior) {
        duplicate |= exact_chunks[row * exact_width + prior] == chunk;
      }
      if (!duplicate) {
        deduplicated_exact_chunks[row * exact_width + deduplicated_exact_count] = chunk;
        ++deduplicated_exact_count;
      }
    }
  }
  __syncthreads();

  const bool generation_matches = cached_generations[row] == current_generations[row];
  const int reusable = min(previous_length, max_reuse_chunks);
  for (int index = thread; index < current_length; index += blockDim.x) {
    const int64_t chunk = current_chunks[row * current_width + index];
    int64_t kind = kIgnored;
    bool duplicate = false;
    for (int prior = 0; prior < index; ++prior) {
      duplicate |= current_chunks[row * current_width + prior] == chunk;
    }
    bool exact = false;
    for (int exact_index = 0; exact_index < deduplicated_exact_count; ++exact_index) {
      exact |= deduplicated_exact_chunks[row * exact_width + exact_index] == chunk;
    }
    if (!duplicate && !exact) {
      bool hit = false;
      if (generation_matches) {
        for (int previous_index = 0; previous_index < reusable; ++previous_index) {
          hit |= previous_chunks[row * previous_width + previous_index] == chunk;
        }
      }
      if (hit) {
        kind = kHit;
      } else {
        kind = kMiss;
      }
    }
    const int64_t plan_offset = (static_cast<int64_t>(row) * current_width + index) * 3;
    plan[plan_offset] = kind;
    plan[plan_offset + 1] = chunk;
    plan[plan_offset + 2] = -1;
  }
  __syncthreads();

  for (int index = thread; index < current_length; index += blockDim.x) {
    const int64_t plan_offset = (static_cast<int64_t>(row) * current_width + index) * 3;
    if (plan[plan_offset] == kMiss) {
      int miss_rank = 0;
      for (int prior = 0; prior < index; ++prior) {
        const int64_t prior_offset = (static_cast<int64_t>(row) * current_width + prior) * 3;
        miss_rank += plan[prior_offset] == kMiss;
      }
      plan[plan_offset + 2] = static_cast<int64_t>(miss_rank) * chunk_size;
    }
  }
  __syncthreads();

  if (thread == 0) {
    int hit_count = 0;
    int miss_count = 0;
    for (int index = 0; index < current_length; ++index) {
      const int64_t kind = plan[(static_cast<int64_t>(row) * current_width + index) * 3];
      hit_count += kind == kHit;
      miss_count += kind == kMiss;
    }
    counts[row * 3] = hit_count;
    counts[row * 3 + 1] = miss_count;
    counts[row * 3 + 2] = deduplicated_exact_count;
  }
}

void check_int64_matrix(const at::Tensor& tensor, const char* name) {
  CHECK_INPUT(tensor);
  TORCH_CHECK(tensor.dim() == 2, name, " must be a 2D tensor");
  TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Long, name, " must use int64");
}

void check_int32_vector(const at::Tensor& tensor, const char* name, int64_t rows) {
  CHECK_INPUT(tensor);
  TORCH_CHECK(tensor.dim() == 1 && tensor.numel() == rows, name, " must have shape [rows]");
  TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Int, name, " must use int32");
}

void check_int64_vector(const at::Tensor& tensor, const char* name, int64_t rows) {
  CHECK_INPUT(tensor);
  TORCH_CHECK(tensor.dim() == 1 && tensor.numel() == rows, name, " must have shape [rows]");
  TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Long, name, " must use int64");
}

}  // namespace

void shadowkv_plan_reuse(
    const at::Tensor& previous_chunks,
    const at::Tensor& previous_lengths,
    const at::Tensor& current_chunks,
    const at::Tensor& current_lengths,
    const at::Tensor& exact_chunks,
    const at::Tensor& exact_lengths,
    const at::Tensor& cached_generations,
    const at::Tensor& current_generations,
    int64_t max_reuse_chunks,
    int64_t chunk_size,
    at::Tensor& plan,
    at::Tensor& deduplicated_exact_chunks,
    at::Tensor& counts,
    at::Tensor& error_codes) {
  check_int64_matrix(previous_chunks, "previous_chunks");
  check_int64_matrix(current_chunks, "current_chunks");
  check_int64_matrix(exact_chunks, "exact_chunks");
  const int64_t rows = current_chunks.size(0);
  TORCH_CHECK(previous_chunks.size(0) == rows && exact_chunks.size(0) == rows, "all chunk tensors need equal rows");
  check_int32_vector(previous_lengths, "previous_lengths", rows);
  check_int32_vector(current_lengths, "current_lengths", rows);
  check_int32_vector(exact_lengths, "exact_lengths", rows);
  check_int64_vector(cached_generations, "cached_generations", rows);
  check_int64_vector(current_generations, "current_generations", rows);
  CHECK_INPUT(plan);
  CHECK_INPUT(deduplicated_exact_chunks);
  CHECK_INPUT(counts);
  CHECK_INPUT(error_codes);
  TORCH_CHECK(
      plan.scalar_type() == at::ScalarType::Long && plan.dim() == 3 && plan.size(0) == rows &&
          plan.size(1) == current_chunks.size(1) && plan.size(2) == 3,
      "plan must use int64 with shape [rows, current_width, 3]");
  TORCH_CHECK(
      deduplicated_exact_chunks.scalar_type() == at::ScalarType::Long &&
          deduplicated_exact_chunks.sizes() == exact_chunks.sizes(),
      "deduplicated_exact_chunks must match exact_chunks");
  TORCH_CHECK(
      counts.scalar_type() == at::ScalarType::Int && counts.dim() == 2 && counts.size(0) == rows && counts.size(1) == 3,
      "counts must use int32 with shape [rows, 3]");
  TORCH_CHECK(
      error_codes.scalar_type() == at::ScalarType::Int && error_codes.dim() == 1 && error_codes.numel() == rows,
      "error_codes must use int32 with shape [rows]");
  TORCH_CHECK(
      max_reuse_chunks >= 0 && max_reuse_chunks <= previous_chunks.size(1),
      "max_reuse_chunks is out of bounds");
  TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
  TORCH_CHECK(
      previous_chunks.size(1) <= kMaximumChunkWidth && current_chunks.size(1) <= kMaximumChunkWidth,
      "planner chunk widths must not exceed 256");
  TORCH_CHECK(exact_chunks.size(1) <= kMaximumExactWidth, "planner exact width must not exceed 64");
  const auto device = current_chunks.device();
  TORCH_CHECK(
      previous_chunks.device() == device && previous_lengths.device() == device && current_lengths.device() == device &&
          exact_chunks.device() == device && exact_lengths.device() == device &&
          cached_generations.device() == device &&
          current_generations.device() == device && plan.device() == device &&
          deduplicated_exact_chunks.device() == device && counts.device() == device && error_codes.device() == device,
      "all shadowkv_plan_reuse tensors must share one CUDA device");
  TORCH_CHECK(rows <= static_cast<int64_t>(std::numeric_limits<int>::max()), "planner row count exceeds launch bound");

  c10::cuda::CUDAGuard device_guard(device);
  sglang::shadowkv::check_operation_device(
      current_chunks,
      "shadowkv_plan_reuse");
  if (rows == 0) {
    return;
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  shadowkv_plan_reuse_kernel<<<static_cast<int>(rows), kPlannerThreads, 0, stream>>>(
      previous_chunks.data_ptr<int64_t>(),
      previous_lengths.data_ptr<int32_t>(),
      current_chunks.data_ptr<int64_t>(),
      current_lengths.data_ptr<int32_t>(),
      exact_chunks.data_ptr<int64_t>(),
      exact_lengths.data_ptr<int32_t>(),
      cached_generations.data_ptr<int64_t>(),
      current_generations.data_ptr<int64_t>(),
      static_cast<int>(previous_chunks.size(1)),
      static_cast<int>(current_chunks.size(1)),
      static_cast<int>(exact_chunks.size(1)),
      static_cast<int>(max_reuse_chunks),
      static_cast<int>(chunk_size),
      plan.data_ptr<int64_t>(),
      deduplicated_exact_chunks.data_ptr<int64_t>(),
      counts.data_ptr<int32_t>(),
      error_codes.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
