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

#include <cstdint>
#include <limits>
#include <vector>

#include "utils.h"

namespace {

constexpr int kComponents = 2;
constexpr int kChunkSize = 8;
constexpr int kHeadDimension = 128;
constexpr int kThreads = 256;
constexpr int kMaximumSelectedCapacity = 256;
constexpr int kMaximumExactCapacity = 64;
constexpr int kVectorBytes = sizeof(uint4);
constexpr int kChunkBytes = kChunkSize * kHeadDimension * sizeof(at::BFloat16);
constexpr int kVectorsPerChunk = kChunkBytes / kVectorBytes;

static_assert(kChunkBytes % kVectorBytes == 0);
static_assert(kVectorsPerChunk == 128);

void check_b200(const at::Tensor& tensor) {
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, tensor.get_device()));
  TORCH_CHECK(
      properties.major == 10 && properties.minor == 0,
      "shadowkv_publish_device requires NVIDIA B200 compute capability 10.0; found ",
      properties.major,
      ".",
      properties.minor);
}

void check_tensor(const at::Tensor& tensor, const char* name, at::ScalarType dtype, int64_t dimensions) {
  CHECK_INPUT(tensor);
  TORCH_CHECK(tensor.dim() == dimensions, name, " must have ", dimensions, " dimensions");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an invalid dtype");
}

__global__ void shadowkv_publish_device_kernel(
    const int32_t* __restrict__ selected_chunk_ids,
    const int32_t* __restrict__ selected_lengths,
    const int32_t* __restrict__ exact_chunk_ids,
    const int32_t* __restrict__ exact_lengths,
    const int32_t* __restrict__ row_indices,
    const int64_t* __restrict__ row_generations,
    const int32_t* __restrict__ planner_error_codes,
    const uint4* __restrict__ destination_key_values,
    const int64_t* __restrict__ temporal_request_generations,
    const int64_t* __restrict__ temporal_layout_generations,
    int selected_capacity,
    int exact_capacity,
    int request_slots,
    int local_layers,
    int kv_heads,
    int temporal_capacity,
    int32_t* __restrict__ temporal_chunk_ids,
    uint4* __restrict__ temporal_key_values,
    int64_t* __restrict__ temporal_publication_generations,
    uint8_t* __restrict__ temporal_component_validity) {
  __shared__ int32_t retained_ordinals[kMaximumSelectedCapacity];
  __shared__ int32_t retained_chunks[kMaximumSelectedCapacity];
  __shared__ int retained_count;
  __shared__ bool row_valid;
  const int head = blockIdx.x;
  const int thread = threadIdx.x;

  if (thread == 0) {
    retained_count = 0;
    const int selected_length = selected_lengths[head];
    const int exact_length = exact_lengths[head];
    const int request_slot = row_indices[static_cast<int64_t>(head) * 3];
    const int local_layer = row_indices[static_cast<int64_t>(head) * 3 + 1];
    const int kv_head = row_indices[static_cast<int64_t>(head) * 3 + 2];
    const int64_t request_generation = row_generations[static_cast<int64_t>(head) * 3];
    const int64_t layout_generation = row_generations[static_cast<int64_t>(head) * 3 + 1];
    const int64_t plan_generation = row_generations[static_cast<int64_t>(head) * 3 + 2];
    row_valid = planner_error_codes[head] == 0 && selected_length >= 0 &&
        selected_length <= selected_capacity && exact_length >= 0 && exact_length <= exact_capacity &&
        request_slot >= 0 && request_slot < request_slots && local_layer >= 0 && local_layer < local_layers &&
        kv_head == head && request_generation >= 1 && layout_generation >= 1 && plan_generation >= 1 &&
        temporal_request_generations[request_slot] == request_generation &&
        temporal_layout_generations[request_slot] == layout_generation;
    if (row_valid) {
      for (int selected = 0; selected < selected_length; ++selected) {
        const int32_t chunk = selected_chunk_ids[static_cast<int64_t>(head) * selected_capacity + selected];
        if (chunk < 0) {
          row_valid = false;
          break;
        }
        bool duplicate = false;
        for (int retained = 0; retained < retained_count; ++retained) {
          duplicate |= retained_chunks[retained] == chunk;
        }
        bool exact_overlap = false;
        for (int exact = 0; exact < exact_length; ++exact) {
          const int32_t exact_chunk = exact_chunk_ids[static_cast<int64_t>(head) * exact_capacity + exact];
          if (exact_chunk < 0) {
            row_valid = false;
            break;
          }
          exact_overlap |= exact_chunk == chunk;
        }
        if (!row_valid) {
          break;
        }
        if (!duplicate && !exact_overlap && retained_count < temporal_capacity) {
          retained_chunks[retained_count] = chunk;
          retained_ordinals[retained_count] = selected;
          ++retained_count;
        }
      }
    }
  }
  __syncthreads();
  if (!row_valid) {
    return;
  }

  const int request_slot = row_indices[static_cast<int64_t>(head) * 3];
  const int local_layer = row_indices[static_cast<int64_t>(head) * 3 + 1];
  const int64_t plan_generation = row_generations[static_cast<int64_t>(head) * 3 + 2];
  const int64_t temporal_row =
      ((static_cast<int64_t>(request_slot) * local_layers + local_layer) * kv_heads + head) * temporal_capacity;
  const int64_t temporal_component_stride =
      static_cast<int64_t>(request_slots) * local_layers * kv_heads * temporal_capacity;

  for (int retained = thread; retained < temporal_capacity; retained += blockDim.x) {
    temporal_chunk_ids[temporal_row + retained] = -1;
    for (int component = 0; component < kComponents; ++component) {
      const int64_t metadata = static_cast<int64_t>(component) * temporal_component_stride + temporal_row + retained;
      temporal_component_validity[metadata] = 0;
      temporal_publication_generations[metadata] = -1;
    }
  }
  __syncthreads();

  const int64_t copy_vectors =
      static_cast<int64_t>(kComponents) * temporal_capacity * kVectorsPerChunk;
  for (int64_t vector = thread; vector < copy_vectors; vector += blockDim.x) {
    const int vector_in_chunk = vector % kVectorsPerChunk;
    const int64_t retained_component = vector / kVectorsPerChunk;
    const int retained = retained_component % temporal_capacity;
    const int component = retained_component / temporal_capacity;
    const int64_t target_slot =
        static_cast<int64_t>(component) * temporal_component_stride + temporal_row + retained;
    const int64_t target_vector = target_slot * kVectorsPerChunk + vector_in_chunk;
    if (retained < retained_count) {
      const int selected = retained_ordinals[retained];
      const int64_t source_slot =
          (static_cast<int64_t>(component) * kv_heads + head) * selected_capacity + selected;
      temporal_key_values[target_vector] =
          destination_key_values[source_slot * kVectorsPerChunk + vector_in_chunk];
    } else {
      temporal_key_values[target_vector] = uint4{0, 0, 0, 0};
    }
  }
  __syncthreads();

  for (int retained = thread; retained < retained_count; retained += blockDim.x) {
    temporal_chunk_ids[temporal_row + retained] = retained_chunks[retained];
    for (int component = 0; component < kComponents; ++component) {
      const int64_t metadata = static_cast<int64_t>(component) * temporal_component_stride + temporal_row + retained;
      temporal_publication_generations[metadata] = plan_generation;
    }
  }
  __syncthreads();
  for (int retained = thread; retained < retained_count; retained += blockDim.x) {
    for (int component = 0; component < kComponents; ++component) {
      const int64_t metadata = static_cast<int64_t>(component) * temporal_component_stride + temporal_row + retained;
      temporal_component_validity[metadata] = 1;
    }
  }
}

}  // namespace

void shadowkv_publish_device(
    const at::Tensor& selected_chunk_ids,
    const at::Tensor& selected_lengths,
    const at::Tensor& exact_chunk_ids,
    const at::Tensor& exact_lengths,
    const at::Tensor& row_indices,
    const at::Tensor& row_generations,
    const at::Tensor& planner_error_codes,
    const at::Tensor& destination_key_values,
    const at::Tensor& temporal_request_generations,
    const at::Tensor& temporal_layout_generations,
    at::Tensor& temporal_chunk_ids,
    at::Tensor& temporal_key_values,
    at::Tensor& temporal_publication_generations,
    at::Tensor& temporal_component_validity) {
  check_tensor(selected_chunk_ids, "selected_chunk_ids", at::ScalarType::Int, 2);
  check_tensor(selected_lengths, "selected_lengths", at::ScalarType::Int, 1);
  check_tensor(exact_chunk_ids, "exact_chunk_ids", at::ScalarType::Int, 2);
  check_tensor(exact_lengths, "exact_lengths", at::ScalarType::Int, 1);
  check_tensor(row_indices, "row_indices", at::ScalarType::Int, 2);
  check_tensor(row_generations, "row_generations", at::ScalarType::Long, 2);
  check_tensor(planner_error_codes, "planner_error_codes", at::ScalarType::Int, 1);
  check_tensor(destination_key_values, "destination_key_values", at::ScalarType::BFloat16, 5);
  check_tensor(temporal_request_generations, "temporal_request_generations", at::ScalarType::Long, 1);
  check_tensor(temporal_layout_generations, "temporal_layout_generations", at::ScalarType::Long, 1);
  check_tensor(temporal_chunk_ids, "temporal_chunk_ids", at::ScalarType::Int, 4);
  check_tensor(temporal_key_values, "temporal_key_values", at::ScalarType::BFloat16, 7);
  check_tensor(
      temporal_publication_generations, "temporal_publication_generations", at::ScalarType::Long, 5);
  check_tensor(temporal_component_validity, "temporal_component_validity", at::ScalarType::Byte, 5);

  const int64_t kv_heads = selected_chunk_ids.size(0);
  const int64_t selected_capacity = selected_chunk_ids.size(1);
  const int64_t exact_capacity = exact_chunk_ids.size(1);
  const int64_t request_slots = temporal_chunk_ids.size(0);
  const int64_t local_layers = temporal_chunk_ids.size(1);
  const int64_t temporal_capacity = temporal_chunk_ids.size(3);
  TORCH_CHECK(
      kv_heads >= 1 && kv_heads <= std::numeric_limits<int>::max(),
      "publication KV heads exceed CUDA launch bounds");
  TORCH_CHECK(
      selected_capacity >= 1 && selected_capacity <= kMaximumSelectedCapacity,
      "publication selected capacity must be between 1 and 256");
  TORCH_CHECK(exact_capacity <= kMaximumExactCapacity, "publication exact capacity must not exceed 64");
  TORCH_CHECK(
      temporal_capacity >= 0 && temporal_capacity <= selected_capacity,
      "publication temporal capacity exceeds selected capacity");
  TORCH_CHECK(
      selected_lengths.numel() == kv_heads && exact_chunk_ids.size(0) == kv_heads &&
          exact_lengths.numel() == kv_heads && planner_error_codes.numel() == kv_heads,
      "publication row counts differ");
  TORCH_CHECK(
      row_indices.size(0) == kv_heads && row_indices.size(1) == 3 && row_generations.size(0) == kv_heads &&
          row_generations.size(1) == 3,
      "publication row identity must have shape [heads, 3]");
  const std::vector<int64_t> destination_shape = {
      kComponents, kv_heads, selected_capacity, kChunkSize, kHeadDimension};
  TORCH_CHECK(
      destination_key_values.sizes().vec() == destination_shape,
      "destination_key_values must have shape [2, heads, selected, 8, 128]");
  TORCH_CHECK(
      temporal_key_values.size(0) == kComponents && temporal_key_values.size(1) == request_slots &&
          temporal_key_values.size(2) == local_layers && temporal_key_values.size(3) == kv_heads &&
          temporal_key_values.size(4) == temporal_capacity && temporal_key_values.size(5) == kChunkSize &&
          temporal_key_values.size(6) == kHeadDimension,
      "temporal_key_values shape differs from temporal metadata");
  const std::vector<int64_t> temporal_component_shape = {
      kComponents, request_slots, local_layers, kv_heads, temporal_capacity};
  TORCH_CHECK(
      temporal_component_validity.sizes().vec() == temporal_component_shape &&
          temporal_publication_generations.sizes().vec() == temporal_component_shape,
      "temporal component metadata shapes differ");
  TORCH_CHECK(
      temporal_request_generations.numel() == request_slots &&
          temporal_layout_generations.numel() == request_slots,
      "temporal owner generations must have shape [request_slots]");

  const auto device = selected_chunk_ids.device();
  const at::Tensor* tensors[] = {
      &selected_lengths,
      &exact_chunk_ids,
      &exact_lengths,
      &row_indices,
      &row_generations,
      &planner_error_codes,
      &destination_key_values,
      &temporal_request_generations,
      &temporal_layout_generations,
      &temporal_chunk_ids,
      &temporal_key_values,
      &temporal_publication_generations,
      &temporal_component_validity,
  };
  for (const at::Tensor* tensor : tensors) {
    TORCH_CHECK(tensor->device() == device, "all shadowkv_publish_device tensors must share one CUDA device");
  }

  c10::cuda::CUDAGuard device_guard(device);
  check_b200(selected_chunk_ids);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  shadowkv_publish_device_kernel<<<static_cast<int>(kv_heads), kThreads, 0, stream>>>(
      selected_chunk_ids.data_ptr<int32_t>(),
      selected_lengths.data_ptr<int32_t>(),
      exact_chunk_ids.data_ptr<int32_t>(),
      exact_lengths.data_ptr<int32_t>(),
      row_indices.data_ptr<int32_t>(),
      row_generations.data_ptr<int64_t>(),
      planner_error_codes.data_ptr<int32_t>(),
      reinterpret_cast<const uint4*>(destination_key_values.data_ptr<at::BFloat16>()),
      temporal_request_generations.data_ptr<int64_t>(),
      temporal_layout_generations.data_ptr<int64_t>(),
      static_cast<int>(selected_capacity),
      static_cast<int>(exact_capacity),
      static_cast<int>(request_slots),
      static_cast<int>(local_layers),
      static_cast<int>(kv_heads),
      static_cast<int>(temporal_capacity),
      temporal_chunk_ids.data_ptr<int32_t>(),
      reinterpret_cast<uint4*>(temporal_key_values.data_ptr<at::BFloat16>()),
      temporal_publication_generations.data_ptr<int64_t>(),
      temporal_component_validity.data_ptr<uint8_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
