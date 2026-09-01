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

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

#include <cstdint>
#include <initializer_list>
#include <limits>

#include "utils.h"

namespace {

constexpr int8_t kInactive = -1;
constexpr int8_t kHit = 1;
constexpr int8_t kMiss = 2;
constexpr int32_t kInvalidLength = 1;
constexpr int32_t kInvalidActiveChunk = 2;
constexpr int32_t kInvalidRowIdentity = 3;
constexpr int32_t kInvalidPlanSlot = 4;
constexpr int32_t kInvalidComponentValidity = 5;
constexpr int32_t kDuplicateTemporalChunk = 6;
constexpr int32_t kInvalidPublicationGeneration = 7;
constexpr int kComponents = 2;
constexpr int kPlannerThreads = 256;
constexpr int kMaximumSelectedCapacity = 256;
constexpr int kMaximumExactCapacity = 64;

void check_b200(const at::Tensor& tensor) {
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, tensor.get_device()));
  TORCH_CHECK(
      properties.major == 10 && properties.minor == 0,
      "shadowkv_plan_device requires NVIDIA B200 compute capability 10.0; found ",
      properties.major,
      ".",
      properties.minor);
}

void check_tensor(const at::Tensor& tensor, const char* name, at::ScalarType dtype, int64_t dimensions) {
  CHECK_INPUT(tensor);
  TORCH_CHECK(tensor.dim() == dimensions, name, " must have ", dimensions, " dimensions");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an invalid dtype");
}

bool logical_slots_fit_int32(std::initializer_list<int64_t> dimensions) {
  constexpr uint64_t limit = static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) + 1;
  uint64_t product = 1;
  for (const int64_t dimension : dimensions) {
    if (dimension < 0 || static_cast<uint64_t>(dimension) > limit / product) {
      return false;
    }
    product *= static_cast<uint64_t>(dimension);
  }
  return true;
}

struct DevicePlanDimensions {
  int64_t rows;
  int64_t selected_capacity;
  int64_t exact_capacity;
  int64_t request_slots;
  int64_t local_layers;
  int64_t kv_heads;
  int64_t temporal_capacity;
};

DevicePlanDimensions validate_device_plan_tensors(
    const at::Tensor& selected_chunk_ids,
    const at::Tensor& selected_lengths,
    const at::Tensor& exact_chunk_ids,
    const at::Tensor& exact_lengths,
    const at::Tensor& temporal_chunk_ids,
    const at::Tensor& temporal_component_validity,
    const at::Tensor& temporal_publication_generations,
    const at::Tensor& temporal_request_generations,
    const at::Tensor& temporal_layout_generations,
    const at::Tensor& row_indices,
    const at::Tensor& row_generations,
    const at::Tensor& plan_slots,
    int64_t plan_capacity,
    const at::Tensor& component_kinds,
    const at::Tensor& source_slots,
    const at::Tensor& destination_slots,
    const at::Tensor& miss_ordinals,
    const at::Tensor& counts,
    const at::Tensor& error_codes) {
  check_tensor(selected_chunk_ids, "selected_chunk_ids", at::ScalarType::Int, 2);
  check_tensor(selected_lengths, "selected_lengths", at::ScalarType::Int, 1);
  check_tensor(exact_chunk_ids, "exact_chunk_ids", at::ScalarType::Int, 2);
  check_tensor(exact_lengths, "exact_lengths", at::ScalarType::Int, 1);
  check_tensor(temporal_chunk_ids, "temporal_chunk_ids", at::ScalarType::Int, 4);
  check_tensor(
      temporal_component_validity, "temporal_component_validity", at::ScalarType::Byte, 5);
  check_tensor(
      temporal_publication_generations, "temporal_publication_generations", at::ScalarType::Long, 5);
  check_tensor(temporal_request_generations, "temporal_request_generations", at::ScalarType::Long, 1);
  check_tensor(temporal_layout_generations, "temporal_layout_generations", at::ScalarType::Long, 1);
  check_tensor(row_indices, "row_indices", at::ScalarType::Int, 2);
  check_tensor(row_generations, "row_generations", at::ScalarType::Long, 2);
  check_tensor(plan_slots, "plan_slots", at::ScalarType::Int, 1);
  check_tensor(component_kinds, "component_kinds", at::ScalarType::Char, 3);
  check_tensor(source_slots, "source_slots", at::ScalarType::Int, 3);
  check_tensor(destination_slots, "destination_slots", at::ScalarType::Int, 3);
  check_tensor(miss_ordinals, "miss_ordinals", at::ScalarType::Int, 3);
  check_tensor(counts, "counts", at::ScalarType::Int, 3);
  check_tensor(error_codes, "error_codes", at::ScalarType::Int, 1);

  const int64_t rows = selected_chunk_ids.size(0);
  const int64_t selected_capacity = selected_chunk_ids.size(1);
  const int64_t exact_capacity = exact_chunk_ids.size(1);
  const int64_t request_slots = temporal_chunk_ids.size(0);
  const int64_t local_layers = temporal_chunk_ids.size(1);
  const int64_t kv_heads = temporal_chunk_ids.size(2);
  const int64_t temporal_capacity = temporal_chunk_ids.size(3);
  TORCH_CHECK(
      selected_capacity >= 1 && selected_capacity <= kMaximumSelectedCapacity,
      "selected capacity must be between 1 and 256");
  TORCH_CHECK(exact_capacity <= kMaximumExactCapacity, "exact capacity must not exceed 64");
  TORCH_CHECK(
      request_slots >= 1 && local_layers >= 1 && kv_heads >= 1,
      "temporal owner dimensions must be positive");
  TORCH_CHECK(
      temporal_capacity <= selected_capacity,
      "temporal capacity must not exceed selected capacity");
  TORCH_CHECK(plan_capacity >= 1, "plan_capacity must be positive");
  TORCH_CHECK(
      rows <= std::numeric_limits<int>::max() && selected_capacity <= std::numeric_limits<int>::max() &&
          exact_capacity <= std::numeric_limits<int>::max() &&
          request_slots <= std::numeric_limits<int>::max() && local_layers <= std::numeric_limits<int>::max() &&
          kv_heads <= std::numeric_limits<int>::max() && temporal_capacity <= std::numeric_limits<int>::max() &&
          plan_capacity <= std::numeric_limits<int>::max(),
      "device-plan dimensions exceed CUDA launch bounds");
  TORCH_CHECK(
      logical_slots_fit_int32({kComponents, plan_capacity, kv_heads, selected_capacity}),
      "logical destination slots exceed int32");
  TORCH_CHECK(
      logical_slots_fit_int32(
          {kComponents, request_slots, local_layers, kv_heads, temporal_capacity}),
      "logical source slots exceed int32");

  TORCH_CHECK(
      selected_lengths.numel() == rows && exact_chunk_ids.size(0) == rows && exact_lengths.numel() == rows &&
          plan_slots.numel() == rows,
      "selected, exact, and plan-slot row counts differ");
  TORCH_CHECK(
      row_indices.size(0) == rows && row_indices.size(1) == 3 && row_generations.size(0) == rows &&
          row_generations.size(1) == 3,
      "row identity tensors must have shape [rows, 3]");
  TORCH_CHECK(
      temporal_component_validity.size(0) == kComponents &&
          temporal_component_validity.size(1) == request_slots &&
          temporal_component_validity.size(2) == local_layers && temporal_component_validity.size(3) == kv_heads &&
          temporal_component_validity.size(4) == temporal_capacity &&
          temporal_publication_generations.sizes() == temporal_component_validity.sizes(),
      "temporal component tensors have incompatible shapes");
  TORCH_CHECK(
      temporal_request_generations.numel() == request_slots &&
          temporal_layout_generations.numel() == request_slots,
      "temporal owner generations must have shape [request_slots]");
  TORCH_CHECK(
      component_kinds.size(0) == kComponents && component_kinds.size(1) == rows &&
          component_kinds.size(2) == selected_capacity && source_slots.sizes() == component_kinds.sizes() &&
          destination_slots.sizes() == component_kinds.sizes() && miss_ordinals.sizes() == component_kinds.sizes(),
      "device-plan component outputs must have shape [2, rows, selected_capacity]");
  TORCH_CHECK(
      counts.size(0) == kComponents && counts.size(1) == rows && counts.size(2) == 2,
      "counts must have shape [2, rows, 2]");
  TORCH_CHECK(error_codes.numel() == rows, "error_codes must have shape [rows]");

  const auto device = selected_chunk_ids.device();
  const at::Tensor* tensors[] = {
      &selected_lengths,
      &exact_chunk_ids,
      &exact_lengths,
      &temporal_chunk_ids,
      &temporal_component_validity,
      &temporal_publication_generations,
      &temporal_request_generations,
      &temporal_layout_generations,
      &row_indices,
      &row_generations,
      &plan_slots,
      &component_kinds,
      &source_slots,
      &destination_slots,
      &miss_ordinals,
      &counts,
      &error_codes,
  };
  for (const at::Tensor* tensor : tensors) {
    TORCH_CHECK(tensor->device() == device, "all shadowkv_plan_device tensors must share one CUDA device");
  }
  return {
      rows,
      selected_capacity,
      exact_capacity,
      request_slots,
      local_layers,
      kv_heads,
      temporal_capacity,
  };
}

__device__ __forceinline__ void record_error(int32_t& error, int32_t candidate) {
  if (candidate != 0 && (error == 0 || candidate < error)) {
    error = candidate;
  }
}

__global__ void shadowkv_plan_device_kernel(
    const int32_t* __restrict__ selected_chunk_ids,
    const int32_t* __restrict__ selected_lengths,
    const int32_t* __restrict__ exact_chunk_ids,
    const int32_t* __restrict__ exact_lengths,
    const int32_t* __restrict__ temporal_chunk_ids,
    const uint8_t* __restrict__ temporal_component_validity,
    const int64_t* __restrict__ temporal_publication_generations,
    const int64_t* __restrict__ temporal_request_generations,
    const int64_t* __restrict__ temporal_layout_generations,
    const int32_t* __restrict__ row_indices,
    const int64_t* __restrict__ row_generations,
    const int32_t* __restrict__ plan_slots,
    int rows,
    int selected_capacity,
    int exact_capacity,
    int request_slots,
    int local_layers,
    int kv_heads,
    int temporal_capacity,
    int plan_capacity,
    int8_t* __restrict__ component_kinds,
    int32_t* __restrict__ source_slots,
    int32_t* __restrict__ destination_slots,
    int32_t* __restrict__ miss_ordinals,
    int32_t* __restrict__ counts,
    int32_t* __restrict__ error_codes) {
  const int row = blockIdx.x;
  const int thread = threadIdx.x;
  const int64_t component_stride = static_cast<int64_t>(rows) * selected_capacity;

  for (int index = thread; index < kComponents * selected_capacity; index += blockDim.x) {
    const int component = index / selected_capacity;
    const int selected = index % selected_capacity;
    const int64_t offset = static_cast<int64_t>(component) * component_stride +
        static_cast<int64_t>(row) * selected_capacity + selected;
    component_kinds[offset] = kInactive;
    source_slots[offset] = -1;
    destination_slots[offset] = -1;
    miss_ordinals[offset] = -1;
  }
  for (int index = thread; index < kComponents * 2; index += blockDim.x) {
    const int component = index / 2;
    const int count_kind = index % 2;
    counts[(static_cast<int64_t>(component) * rows + row) * 2 + count_kind] = 0;
  }
  if (thread == 0) {
    int32_t error = 0;
    const int selected_length = selected_lengths[row];
    const int exact_length = exact_lengths[row];
    if (selected_length < 0 || selected_length > selected_capacity || exact_length < 0 ||
        exact_length > exact_capacity) {
      record_error(error, kInvalidLength);
    }

    if (error != kInvalidLength) {
      for (int selected = 0; selected < selected_length; ++selected) {
        if (selected_chunk_ids[static_cast<int64_t>(row) * selected_capacity + selected] < 0) {
          record_error(error, kInvalidActiveChunk);
        }
      }
      for (int exact = 0; exact < exact_length; ++exact) {
        if (exact_chunk_ids[static_cast<int64_t>(row) * exact_capacity + exact] < 0) {
          record_error(error, kInvalidActiveChunk);
        }
      }
    }

    const int request_slot = row_indices[static_cast<int64_t>(row) * 3];
    const int local_layer = row_indices[static_cast<int64_t>(row) * 3 + 1];
    const int kv_head = row_indices[static_cast<int64_t>(row) * 3 + 2];
    const int64_t request_generation = row_generations[static_cast<int64_t>(row) * 3];
    const int64_t layout_generation = row_generations[static_cast<int64_t>(row) * 3 + 1];
    const int64_t plan_generation = row_generations[static_cast<int64_t>(row) * 3 + 2];
    const bool identity_valid = request_slot >= 0 && request_slot < request_slots && local_layer >= 0 &&
        local_layer < local_layers && kv_head >= 0 && kv_head < kv_heads && request_generation >= 1 &&
        layout_generation >= 1 && plan_generation >= 1;
    if (!identity_valid) {
      record_error(error, kInvalidRowIdentity);
    }
    const int plan_slot = plan_slots[row];
    if (plan_slot < 0 || plan_slot >= plan_capacity) {
      record_error(error, kInvalidPlanSlot);
    }

    bool owner_compatible = false;
    int64_t temporal_base = 0;
    int64_t temporal_component_stride = 0;
    if (identity_valid) {
      owner_compatible = temporal_request_generations[request_slot] == request_generation &&
          temporal_layout_generations[request_slot] == layout_generation;
      temporal_base =
          ((static_cast<int64_t>(request_slot) * local_layers + local_layer) * kv_heads + kv_head) *
          temporal_capacity;
      temporal_component_stride =
          static_cast<int64_t>(request_slots) * local_layers * kv_heads * temporal_capacity;
    }

    if (owner_compatible) {
      for (int temporal = 0; temporal < temporal_capacity; ++temporal) {
        if (temporal_chunk_ids[temporal_base + temporal] < -1) {
          record_error(error, kInvalidActiveChunk);
        }
      }
      for (int component = 0; component < kComponents; ++component) {
        for (int temporal = 0; temporal < temporal_capacity; ++temporal) {
          const int64_t offset =
              static_cast<int64_t>(component) * temporal_component_stride + temporal_base + temporal;
          const uint8_t valid = temporal_component_validity[offset];
          if (valid > 1) {
            record_error(error, kInvalidComponentValidity);
          }
        }
      }
      for (int temporal = 0; temporal < temporal_capacity; ++temporal) {
        const int32_t chunk = temporal_chunk_ids[temporal_base + temporal];
        if (chunk < 0) {
          continue;
        }
        for (int prior = 0; prior < temporal; ++prior) {
          if (temporal_chunk_ids[temporal_base + prior] == chunk) {
            record_error(error, kDuplicateTemporalChunk);
          }
        }
      }
      for (int component = 0; component < kComponents; ++component) {
        for (int temporal = 0; temporal < temporal_capacity; ++temporal) {
          const int64_t offset =
              static_cast<int64_t>(component) * temporal_component_stride + temporal_base + temporal;
          const int32_t chunk = temporal_chunk_ids[temporal_base + temporal];
          const uint8_t valid = temporal_component_validity[offset];
          const int64_t publication = temporal_publication_generations[offset];
          if (publication < -1 ||
              (valid == 1 && (chunk < 0 || publication < 0 || publication >= plan_generation))) {
            record_error(error, kInvalidPublicationGeneration);
          }
        }
      }
    }
    error_codes[row] = error;
  }
  __syncthreads();
  if (error_codes[row] != 0 || thread != 0) {
    return;
  }

  const int selected_length = selected_lengths[row];
  const int exact_length = exact_lengths[row];
  const int request_slot = row_indices[static_cast<int64_t>(row) * 3];
  const int local_layer = row_indices[static_cast<int64_t>(row) * 3 + 1];
  const int kv_head = row_indices[static_cast<int64_t>(row) * 3 + 2];
  const int64_t request_generation = row_generations[static_cast<int64_t>(row) * 3];
  const int64_t layout_generation = row_generations[static_cast<int64_t>(row) * 3 + 1];
  const int plan_slot = plan_slots[row];
  const bool owner_compatible = temporal_request_generations[request_slot] == request_generation &&
      temporal_layout_generations[request_slot] == layout_generation;
  const int64_t temporal_base =
      ((static_cast<int64_t>(request_slot) * local_layers + local_layer) * kv_heads + kv_head) * temporal_capacity;
  const int64_t temporal_component_stride =
      static_cast<int64_t>(request_slots) * local_layers * kv_heads * temporal_capacity;
  int hit_counts[kComponents] = {0, 0};
  int miss_counts[kComponents] = {0, 0};

  for (int selected = 0; selected < selected_length; ++selected) {
    const int32_t chunk = selected_chunk_ids[static_cast<int64_t>(row) * selected_capacity + selected];
    bool duplicate = false;
    for (int prior = 0; prior < selected; ++prior) {
      duplicate |= selected_chunk_ids[static_cast<int64_t>(row) * selected_capacity + prior] == chunk;
    }
    bool exact_overlap = false;
    for (int exact = 0; exact < exact_length; ++exact) {
      exact_overlap |= exact_chunk_ids[static_cast<int64_t>(row) * exact_capacity + exact] == chunk;
    }
    if (duplicate || exact_overlap) {
      continue;
    }

    for (int component = 0; component < kComponents; ++component) {
      const int64_t plan_offset = static_cast<int64_t>(component) * component_stride +
          static_cast<int64_t>(row) * selected_capacity + selected;
      const int64_t destination =
          ((static_cast<int64_t>(component) * plan_capacity + plan_slot) * kv_heads + kv_head) *
              selected_capacity +
          selected;
      destination_slots[plan_offset] = static_cast<int32_t>(destination);
      int temporal_match = -1;
      if (owner_compatible) {
        for (int temporal = 0; temporal < temporal_capacity; ++temporal) {
          if (temporal_chunk_ids[temporal_base + temporal] == chunk) {
            temporal_match = temporal;
            break;
          }
        }
      }
      const int64_t validity_offset = static_cast<int64_t>(component) * temporal_component_stride +
          temporal_base + temporal_match;
      const bool hit = temporal_match >= 0 && temporal_component_validity[validity_offset] == 1;
      if (hit) {
        component_kinds[plan_offset] = kHit;
        const int64_t source =
            ((((static_cast<int64_t>(component) * request_slots + request_slot) * local_layers + local_layer) *
                  kv_heads +
              kv_head) *
                 temporal_capacity) +
            temporal_match;
        source_slots[plan_offset] = static_cast<int32_t>(source);
        ++hit_counts[component];
      } else {
        component_kinds[plan_offset] = kMiss;
        miss_ordinals[plan_offset] = miss_counts[component]++;
      }
    }
  }
  for (int component = 0; component < kComponents; ++component) {
    const int64_t count_offset = (static_cast<int64_t>(component) * rows + row) * 2;
    counts[count_offset] = hit_counts[component];
    counts[count_offset + 1] = miss_counts[component];
  }
}

__device__ __forceinline__ void record_parallel_error(int32_t* error, int32_t candidate) {
  if (candidate != 0) {
    atomicMin(error, candidate);
  }
}

__global__ void shadowkv_plan_device_v2_kernel(
    const int32_t* __restrict__ selected_chunk_ids,
    const int32_t* __restrict__ selected_lengths,
    const int32_t* __restrict__ exact_chunk_ids,
    const int32_t* __restrict__ exact_lengths,
    const int32_t* __restrict__ temporal_chunk_ids,
    const uint8_t* __restrict__ temporal_component_validity,
    const int64_t* __restrict__ temporal_publication_generations,
    const int64_t* __restrict__ temporal_request_generations,
    const int64_t* __restrict__ temporal_layout_generations,
    const int32_t* __restrict__ row_indices,
    const int64_t* __restrict__ row_generations,
    const int32_t* __restrict__ plan_slots,
    int rows,
    int selected_capacity,
    int exact_capacity,
    int request_slots,
    int local_layers,
    int kv_heads,
    int temporal_capacity,
    int plan_capacity,
    int8_t* __restrict__ component_kinds,
    int32_t* __restrict__ source_slots,
    int32_t* __restrict__ destination_slots,
    int32_t* __restrict__ miss_ordinals,
    int32_t* __restrict__ counts,
    int32_t* __restrict__ error_codes,
    int32_t* __restrict__ value_miss_chunk_ids,
    int32_t* __restrict__ value_miss_lengths) {
  const int row = blockIdx.x;
  const int thread = threadIdx.x;
  const int64_t component_stride = static_cast<int64_t>(rows) * selected_capacity;
  constexpr int32_t kNoError = std::numeric_limits<int32_t>::max();

  using BlockReduce = cub::BlockReduce<int, kPlannerThreads>;
  using BlockScan = cub::BlockScan<int, kPlannerThreads>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ int32_t row_error;
  __shared__ int selected_length_shared;
  __shared__ int exact_length_shared;
  __shared__ int request_slot_shared;
  __shared__ int local_layer_shared;
  __shared__ int kv_head_shared;
  __shared__ int plan_slot_shared;
  __shared__ int64_t request_generation_shared;
  __shared__ int64_t layout_generation_shared;
  __shared__ int64_t plan_generation_shared;
  __shared__ int identity_valid_shared;
  __shared__ int owner_compatible_shared;
  __shared__ int effective_count_shared;
  __shared__ int miss_counts_shared[kComponents];

  for (int index = thread; index < kComponents * selected_capacity; index += blockDim.x) {
    const int component = index / selected_capacity;
    const int selected = index % selected_capacity;
    const int64_t offset = static_cast<int64_t>(component) * component_stride +
        static_cast<int64_t>(row) * selected_capacity + selected;
    component_kinds[offset] = kInactive;
    source_slots[offset] = -1;
    destination_slots[offset] = -1;
    miss_ordinals[offset] = -1;
  }
  if (thread < selected_capacity) {
    value_miss_chunk_ids[static_cast<int64_t>(row) * selected_capacity + thread] = -1;
  }
  for (int index = thread; index < kComponents * 2; index += blockDim.x) {
    const int component = index / 2;
    const int count_kind = index % 2;
    counts[(static_cast<int64_t>(component) * rows + row) * 2 + count_kind] = 0;
  }
  if (thread == 0) {
    row_error = kNoError;
    value_miss_lengths[row] = 0;
    selected_length_shared = selected_lengths[row];
    exact_length_shared = exact_lengths[row];
    request_slot_shared = row_indices[static_cast<int64_t>(row) * 3];
    local_layer_shared = row_indices[static_cast<int64_t>(row) * 3 + 1];
    kv_head_shared = row_indices[static_cast<int64_t>(row) * 3 + 2];
    request_generation_shared = row_generations[static_cast<int64_t>(row) * 3];
    layout_generation_shared = row_generations[static_cast<int64_t>(row) * 3 + 1];
    plan_generation_shared = row_generations[static_cast<int64_t>(row) * 3 + 2];
    plan_slot_shared = plan_slots[row];
    const bool lengths_valid = selected_length_shared >= 0 && selected_length_shared <= selected_capacity &&
        exact_length_shared >= 0 && exact_length_shared <= exact_capacity;
    if (!lengths_valid) {
      record_parallel_error(&row_error, kInvalidLength);
    }
    identity_valid_shared = request_slot_shared >= 0 && request_slot_shared < request_slots &&
        local_layer_shared >= 0 && local_layer_shared < local_layers && kv_head_shared >= 0 &&
        kv_head_shared < kv_heads && request_generation_shared >= 1 && layout_generation_shared >= 1 &&
        plan_generation_shared >= 1;
    if (!identity_valid_shared) {
      record_parallel_error(&row_error, kInvalidRowIdentity);
    }
    if (plan_slot_shared < 0 || plan_slot_shared >= plan_capacity) {
      record_parallel_error(&row_error, kInvalidPlanSlot);
    }
    owner_compatible_shared = identity_valid_shared &&
        temporal_request_generations[request_slot_shared] == request_generation_shared &&
        temporal_layout_generations[request_slot_shared] == layout_generation_shared;
  }
  __syncthreads();

  const bool lengths_valid = selected_length_shared >= 0 && selected_length_shared <= selected_capacity &&
      exact_length_shared >= 0 && exact_length_shared <= exact_capacity;
  if (lengths_valid && thread < selected_length_shared &&
      selected_chunk_ids[static_cast<int64_t>(row) * selected_capacity + thread] < 0) {
    record_parallel_error(&row_error, kInvalidActiveChunk);
  }
  if (lengths_valid && thread < exact_length_shared &&
      exact_chunk_ids[static_cast<int64_t>(row) * exact_capacity + thread] < 0) {
    record_parallel_error(&row_error, kInvalidActiveChunk);
  }

  int64_t temporal_base = 0;
  const int64_t temporal_component_stride =
      static_cast<int64_t>(request_slots) * local_layers * kv_heads * temporal_capacity;
  if (identity_valid_shared) {
    temporal_base =
        ((static_cast<int64_t>(request_slot_shared) * local_layers + local_layer_shared) * kv_heads +
         kv_head_shared) *
        temporal_capacity;
  }
  if (owner_compatible_shared && thread < temporal_capacity) {
    const int32_t chunk = temporal_chunk_ids[temporal_base + thread];
    if (chunk < -1) {
      record_parallel_error(&row_error, kInvalidActiveChunk);
    }
    for (int prior = 0; prior < thread; ++prior) {
      if (chunk >= 0 && temporal_chunk_ids[temporal_base + prior] == chunk) {
        record_parallel_error(&row_error, kDuplicateTemporalChunk);
      }
    }
    for (int component = 0; component < kComponents; ++component) {
      const int64_t offset =
          static_cast<int64_t>(component) * temporal_component_stride + temporal_base + thread;
      const uint8_t valid = temporal_component_validity[offset];
      const int64_t publication = temporal_publication_generations[offset];
      if (valid > 1) {
        record_parallel_error(&row_error, kInvalidComponentValidity);
      }
      if (publication < -1 ||
          (valid == 1 && (chunk < 0 || publication < 0 || publication >= plan_generation_shared))) {
        record_parallel_error(&row_error, kInvalidPublicationGeneration);
      }
    }
  }
  __syncthreads();
  if (thread == 0) {
    error_codes[row] = row_error == kNoError ? 0 : row_error;
  }
  __syncthreads();
  if (error_codes[row] != 0) {
    return;
  }

  const bool selected_active = thread < selected_length_shared;
  int32_t chunk = -1;
  bool duplicate = false;
  bool exact_overlap = false;
  if (selected_active) {
    chunk = selected_chunk_ids[static_cast<int64_t>(row) * selected_capacity + thread];
    for (int prior = 0; prior < thread; ++prior) {
      duplicate |= selected_chunk_ids[static_cast<int64_t>(row) * selected_capacity + prior] == chunk;
    }
    for (int exact = 0; exact < exact_length_shared; ++exact) {
      exact_overlap |= exact_chunk_ids[static_cast<int64_t>(row) * exact_capacity + exact] == chunk;
    }
  }
  const bool effective = selected_active && !duplicate && !exact_overlap;
  int temporal_match = -1;
  if (effective && owner_compatible_shared) {
    for (int temporal = 0; temporal < temporal_capacity; ++temporal) {
      if (temporal_chunk_ids[temporal_base + temporal] == chunk) {
        temporal_match = temporal;
        break;
      }
    }
  }

  int miss_flags[kComponents] = {0, 0};
  if (effective) {
    for (int component = 0; component < kComponents; ++component) {
      const int64_t plan_offset = static_cast<int64_t>(component) * component_stride +
          static_cast<int64_t>(row) * selected_capacity + thread;
      const int64_t destination =
          ((static_cast<int64_t>(component) * plan_capacity + plan_slot_shared) * kv_heads + kv_head_shared) *
              selected_capacity +
          thread;
      destination_slots[plan_offset] = static_cast<int32_t>(destination);
      const bool hit = temporal_match >= 0 &&
          temporal_component_validity[static_cast<int64_t>(component) * temporal_component_stride + temporal_base +
                                      temporal_match] == 1;
      if (hit) {
        component_kinds[plan_offset] = kHit;
        const int64_t source =
            ((((static_cast<int64_t>(component) * request_slots + request_slot_shared) * local_layers +
               local_layer_shared) *
                  kv_heads +
              kv_head_shared) *
                 temporal_capacity) +
            temporal_match;
        source_slots[plan_offset] = static_cast<int32_t>(source);
      } else {
        component_kinds[plan_offset] = kMiss;
        miss_flags[component] = 1;
      }
    }
  }

  const int effective_count = BlockReduce(reduce_storage).Sum(effective ? 1 : 0);
  if (thread == 0) {
    effective_count_shared = effective_count;
  }
  __syncthreads();

  for (int component = 0; component < kComponents; ++component) {
    int miss_prefix = 0;
    BlockScan(scan_storage).InclusiveSum(miss_flags[component], miss_prefix);
    if (miss_flags[component] != 0) {
      const int64_t plan_offset = static_cast<int64_t>(component) * component_stride +
          static_cast<int64_t>(row) * selected_capacity + thread;
      miss_ordinals[plan_offset] = miss_prefix - 1;
      if (component == 1) {
        value_miss_chunk_ids[static_cast<int64_t>(row) * selected_capacity + miss_prefix - 1] = chunk;
      }
    }
    if (thread == kPlannerThreads - 1) {
      miss_counts_shared[component] = miss_prefix;
    }
    __syncthreads();
    if (thread == 0) {
      const int64_t count_offset = (static_cast<int64_t>(component) * rows + row) * 2;
      counts[count_offset] = effective_count_shared - miss_counts_shared[component];
      counts[count_offset + 1] = miss_counts_shared[component];
      if (component == 1) {
        value_miss_lengths[row] = miss_counts_shared[component];
      }
    }
    __syncthreads();
  }
}

}  // namespace

void shadowkv_plan_device(
    const at::Tensor& selected_chunk_ids,
    const at::Tensor& selected_lengths,
    const at::Tensor& exact_chunk_ids,
    const at::Tensor& exact_lengths,
    const at::Tensor& temporal_chunk_ids,
    const at::Tensor& temporal_component_validity,
    const at::Tensor& temporal_publication_generations,
    const at::Tensor& temporal_request_generations,
    const at::Tensor& temporal_layout_generations,
    const at::Tensor& row_indices,
    const at::Tensor& row_generations,
    const at::Tensor& plan_slots,
    int64_t plan_capacity,
    at::Tensor& component_kinds,
    at::Tensor& source_slots,
    at::Tensor& destination_slots,
    at::Tensor& miss_ordinals,
    at::Tensor& counts,
    at::Tensor& error_codes) {
  check_tensor(selected_chunk_ids, "selected_chunk_ids", at::ScalarType::Int, 2);
  check_tensor(selected_lengths, "selected_lengths", at::ScalarType::Int, 1);
  check_tensor(exact_chunk_ids, "exact_chunk_ids", at::ScalarType::Int, 2);
  check_tensor(exact_lengths, "exact_lengths", at::ScalarType::Int, 1);
  check_tensor(temporal_chunk_ids, "temporal_chunk_ids", at::ScalarType::Int, 4);
  check_tensor(
      temporal_component_validity, "temporal_component_validity", at::ScalarType::Byte, 5);
  check_tensor(
      temporal_publication_generations, "temporal_publication_generations", at::ScalarType::Long, 5);
  check_tensor(temporal_request_generations, "temporal_request_generations", at::ScalarType::Long, 1);
  check_tensor(temporal_layout_generations, "temporal_layout_generations", at::ScalarType::Long, 1);
  check_tensor(row_indices, "row_indices", at::ScalarType::Int, 2);
  check_tensor(row_generations, "row_generations", at::ScalarType::Long, 2);
  check_tensor(plan_slots, "plan_slots", at::ScalarType::Int, 1);
  check_tensor(component_kinds, "component_kinds", at::ScalarType::Char, 3);
  check_tensor(source_slots, "source_slots", at::ScalarType::Int, 3);
  check_tensor(destination_slots, "destination_slots", at::ScalarType::Int, 3);
  check_tensor(miss_ordinals, "miss_ordinals", at::ScalarType::Int, 3);
  check_tensor(counts, "counts", at::ScalarType::Int, 3);
  check_tensor(error_codes, "error_codes", at::ScalarType::Int, 1);

  const int64_t rows = selected_chunk_ids.size(0);
  const int64_t selected_capacity = selected_chunk_ids.size(1);
  const int64_t exact_capacity = exact_chunk_ids.size(1);
  const int64_t request_slots = temporal_chunk_ids.size(0);
  const int64_t local_layers = temporal_chunk_ids.size(1);
  const int64_t kv_heads = temporal_chunk_ids.size(2);
  const int64_t temporal_capacity = temporal_chunk_ids.size(3);
  TORCH_CHECK(
      selected_capacity >= 1 && selected_capacity <= kMaximumSelectedCapacity,
      "selected capacity must be between 1 and 256");
  TORCH_CHECK(exact_capacity <= kMaximumExactCapacity, "exact capacity must not exceed 64");
  TORCH_CHECK(
      request_slots >= 1 && local_layers >= 1 && kv_heads >= 1,
      "temporal owner dimensions must be positive");
  TORCH_CHECK(
      temporal_capacity <= selected_capacity,
      "temporal capacity must not exceed selected capacity");
  TORCH_CHECK(plan_capacity >= 1, "plan_capacity must be positive");
  TORCH_CHECK(
      rows <= std::numeric_limits<int>::max() && selected_capacity <= std::numeric_limits<int>::max() &&
          exact_capacity <= std::numeric_limits<int>::max() &&
          request_slots <= std::numeric_limits<int>::max() && local_layers <= std::numeric_limits<int>::max() &&
          kv_heads <= std::numeric_limits<int>::max() && temporal_capacity <= std::numeric_limits<int>::max() &&
          plan_capacity <= std::numeric_limits<int>::max(),
      "device-plan dimensions exceed CUDA launch bounds");
  TORCH_CHECK(
      logical_slots_fit_int32({kComponents, plan_capacity, kv_heads, selected_capacity}),
      "logical destination slots exceed int32");
  TORCH_CHECK(
      logical_slots_fit_int32(
          {kComponents, request_slots, local_layers, kv_heads, temporal_capacity}),
      "logical source slots exceed int32");

  TORCH_CHECK(
      selected_lengths.numel() == rows && exact_chunk_ids.size(0) == rows && exact_lengths.numel() == rows &&
          plan_slots.numel() == rows,
      "selected, exact, and plan-slot row counts differ");
  TORCH_CHECK(
      row_indices.size(0) == rows && row_indices.size(1) == 3 && row_generations.size(0) == rows &&
          row_generations.size(1) == 3,
      "row identity tensors must have shape [rows, 3]");
  TORCH_CHECK(
      temporal_component_validity.size(0) == kComponents &&
          temporal_component_validity.size(1) == request_slots &&
          temporal_component_validity.size(2) == local_layers && temporal_component_validity.size(3) == kv_heads &&
          temporal_component_validity.size(4) == temporal_capacity &&
          temporal_publication_generations.sizes() == temporal_component_validity.sizes(),
      "temporal component tensors have incompatible shapes");
  TORCH_CHECK(
      temporal_request_generations.numel() == request_slots &&
          temporal_layout_generations.numel() == request_slots,
      "temporal owner generations must have shape [request_slots]");
  TORCH_CHECK(
      component_kinds.size(0) == kComponents && component_kinds.size(1) == rows &&
          component_kinds.size(2) == selected_capacity && source_slots.sizes() == component_kinds.sizes() &&
          destination_slots.sizes() == component_kinds.sizes() && miss_ordinals.sizes() == component_kinds.sizes(),
      "device-plan component outputs must have shape [2, rows, selected_capacity]");
  TORCH_CHECK(
      counts.size(0) == kComponents && counts.size(1) == rows && counts.size(2) == 2,
      "counts must have shape [2, rows, 2]");
  TORCH_CHECK(error_codes.numel() == rows, "error_codes must have shape [rows]");

  const auto device = selected_chunk_ids.device();
  const at::Tensor* tensors[] = {
      &selected_lengths,
      &exact_chunk_ids,
      &exact_lengths,
      &temporal_chunk_ids,
      &temporal_component_validity,
      &temporal_publication_generations,
      &temporal_request_generations,
      &temporal_layout_generations,
      &row_indices,
      &row_generations,
      &plan_slots,
      &component_kinds,
      &source_slots,
      &destination_slots,
      &miss_ordinals,
      &counts,
      &error_codes,
  };
  for (const at::Tensor* tensor : tensors) {
    TORCH_CHECK(tensor->device() == device, "all shadowkv_plan_device tensors must share one CUDA device");
  }

  const DevicePlanDimensions dimensions{
      rows,
      selected_capacity,
      exact_capacity,
      request_slots,
      local_layers,
      kv_heads,
      temporal_capacity,
  };
  c10::cuda::CUDAGuard device_guard(device);
  check_b200(selected_chunk_ids);
  if (dimensions.rows == 0) {
    return;
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  shadowkv_plan_device_kernel<<<static_cast<int>(dimensions.rows), kPlannerThreads, 0, stream>>>(
      selected_chunk_ids.data_ptr<int32_t>(),
      selected_lengths.data_ptr<int32_t>(),
      exact_chunk_ids.data_ptr<int32_t>(),
      exact_lengths.data_ptr<int32_t>(),
      temporal_chunk_ids.data_ptr<int32_t>(),
      temporal_component_validity.data_ptr<uint8_t>(),
      temporal_publication_generations.data_ptr<int64_t>(),
      temporal_request_generations.data_ptr<int64_t>(),
      temporal_layout_generations.data_ptr<int64_t>(),
      row_indices.data_ptr<int32_t>(),
      row_generations.data_ptr<int64_t>(),
      plan_slots.data_ptr<int32_t>(),
      static_cast<int>(dimensions.rows),
      static_cast<int>(dimensions.selected_capacity),
      static_cast<int>(dimensions.exact_capacity),
      static_cast<int>(dimensions.request_slots),
      static_cast<int>(dimensions.local_layers),
      static_cast<int>(dimensions.kv_heads),
      static_cast<int>(dimensions.temporal_capacity),
      static_cast<int>(plan_capacity),
      component_kinds.data_ptr<int8_t>(),
      source_slots.data_ptr<int32_t>(),
      destination_slots.data_ptr<int32_t>(),
      miss_ordinals.data_ptr<int32_t>(),
      counts.data_ptr<int32_t>(),
      error_codes.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void shadowkv_plan_device_v2(
    const at::Tensor& selected_chunk_ids,
    const at::Tensor& selected_lengths,
    const at::Tensor& exact_chunk_ids,
    const at::Tensor& exact_lengths,
    const at::Tensor& temporal_chunk_ids,
    const at::Tensor& temporal_component_validity,
    const at::Tensor& temporal_publication_generations,
    const at::Tensor& temporal_request_generations,
    const at::Tensor& temporal_layout_generations,
    const at::Tensor& row_indices,
    const at::Tensor& row_generations,
    const at::Tensor& plan_slots,
    int64_t plan_capacity,
    at::Tensor& component_kinds,
    at::Tensor& source_slots,
    at::Tensor& destination_slots,
    at::Tensor& miss_ordinals,
    at::Tensor& counts,
    at::Tensor& error_codes,
    at::Tensor& value_miss_chunk_ids,
    at::Tensor& value_miss_lengths) {
  const DevicePlanDimensions dimensions = validate_device_plan_tensors(
      selected_chunk_ids,
      selected_lengths,
      exact_chunk_ids,
      exact_lengths,
      temporal_chunk_ids,
      temporal_component_validity,
      temporal_publication_generations,
      temporal_request_generations,
      temporal_layout_generations,
      row_indices,
      row_generations,
      plan_slots,
      plan_capacity,
      component_kinds,
      source_slots,
      destination_slots,
      miss_ordinals,
      counts,
      error_codes);
  check_tensor(value_miss_chunk_ids, "value_miss_chunk_ids", at::ScalarType::Int, 2);
  check_tensor(value_miss_lengths, "value_miss_lengths", at::ScalarType::Int, 1);
  TORCH_CHECK(
      value_miss_chunk_ids.size(0) == dimensions.rows &&
          value_miss_chunk_ids.size(1) == dimensions.selected_capacity,
      "value_miss_chunk_ids must have shape [rows, selected_capacity]");
  TORCH_CHECK(
      value_miss_lengths.numel() == dimensions.rows,
      "value_miss_lengths must have shape [rows]");
  const auto device = selected_chunk_ids.device();
  TORCH_CHECK(
      value_miss_chunk_ids.device() == device && value_miss_lengths.device() == device,
      "all shadowkv_plan_device_v2 tensors must share one CUDA device");

  c10::cuda::CUDAGuard device_guard(device);
  check_b200(selected_chunk_ids);
  if (dimensions.rows == 0) {
    return;
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  shadowkv_plan_device_v2_kernel<<<static_cast<int>(dimensions.rows), kPlannerThreads, 0, stream>>>(
      selected_chunk_ids.data_ptr<int32_t>(),
      selected_lengths.data_ptr<int32_t>(),
      exact_chunk_ids.data_ptr<int32_t>(),
      exact_lengths.data_ptr<int32_t>(),
      temporal_chunk_ids.data_ptr<int32_t>(),
      temporal_component_validity.data_ptr<uint8_t>(),
      temporal_publication_generations.data_ptr<int64_t>(),
      temporal_request_generations.data_ptr<int64_t>(),
      temporal_layout_generations.data_ptr<int64_t>(),
      row_indices.data_ptr<int32_t>(),
      row_generations.data_ptr<int64_t>(),
      plan_slots.data_ptr<int32_t>(),
      static_cast<int>(dimensions.rows),
      static_cast<int>(dimensions.selected_capacity),
      static_cast<int>(dimensions.exact_capacity),
      static_cast<int>(dimensions.request_slots),
      static_cast<int>(dimensions.local_layers),
      static_cast<int>(dimensions.kv_heads),
      static_cast<int>(dimensions.temporal_capacity),
      static_cast<int>(plan_capacity),
      component_kinds.data_ptr<int8_t>(),
      source_slots.data_ptr<int32_t>(),
      destination_slots.data_ptr<int32_t>(),
      miss_ordinals.data_ptr<int32_t>(),
      counts.data_ptr<int32_t>(),
      error_codes.data_ptr<int32_t>(),
      value_miss_chunk_ids.data_ptr<int32_t>(),
      value_miss_lengths.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
