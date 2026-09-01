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
#include <initializer_list>
#include <limits>
#include <vector>

#include "utils.h"

namespace {

constexpr int8_t kHit = 1;
constexpr int8_t kMiss = 2;
constexpr int32_t kInvalidPlacementDescriptor = 8;
constexpr int kComponents = 2;
constexpr int kChunkSize = 8;
constexpr int kHeadDimension = 128;
constexpr int kCopyThreads = 128;
constexpr int kVectorBytes = sizeof(uint4);
constexpr int kChunkBytes = kChunkSize * kHeadDimension * sizeof(at::BFloat16);
constexpr int kVectorsPerChunk = kChunkBytes / kVectorBytes;

static_assert(kChunkBytes % kVectorBytes == 0);
static_assert(kVectorsPerChunk == kCopyThreads);

void check_b200(const at::Tensor& tensor) {
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, tensor.get_device()));
  TORCH_CHECK(
      properties.major == 10 && properties.minor == 0,
      "shadowkv_place_device requires NVIDIA B200 compute capability 10.0; found ",
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

void check_mapped_host_device(int device_index) {
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device_index));
  TORCH_CHECK(
      properties.major == 10 && properties.minor == 0,
      "ShadowKV mapped-host access requires NVIDIA B200 compute capability 10.0");
  TORCH_CHECK(properties.canMapHostMemory != 0, "selected CUDA device cannot map host memory");
  TORCH_CHECK(properties.unifiedAddressing != 0, "selected CUDA device lacks unified virtual addressing");
}

__global__ void shadowkv_place_device_kernel(
    const int8_t* __restrict__ component_kinds,
    const int32_t* __restrict__ source_slots,
    const int32_t* __restrict__ destination_slots,
    const int32_t* __restrict__ plan_slots,
    int32_t* __restrict__ planner_error_codes,
    const uint4* __restrict__ temporal_key_values,
    const uint4* __restrict__ compatibility_key_values,
    int kv_heads,
    int selected_capacity,
    int64_t temporal_source_slots,
    int plan_capacity,
    uint4* __restrict__ destination_key_values) {
  const int entry = blockIdx.x;
  const int selected = entry % selected_capacity;
  const int head_component = entry / selected_capacity;
  const int kv_head = head_component % kv_heads;
  const int component = head_component / kv_heads;
  const int64_t plan_offset =
      (static_cast<int64_t>(component) * kv_heads + kv_head) * selected_capacity + selected;
  const int8_t kind = component_kinds[plan_offset];
  const int plan_slot = plan_slots[kv_head];
  const int64_t expected_destination =
      ((static_cast<int64_t>(component) * plan_capacity + plan_slot) * kv_heads + kv_head) *
          selected_capacity +
      selected;
  const int64_t source_slot = source_slots[plan_offset];
  const bool row_valid = planner_error_codes[kv_head] == 0 && plan_slot >= 0 && plan_slot < plan_capacity;
  const bool hit_valid = kind == kHit && source_slot >= 0 && source_slot < temporal_source_slots;
  const bool miss_valid = kind == kMiss && source_slot == -1;
  const bool destination_valid = destination_slots[plan_offset] == expected_destination;
  const bool active = row_valid && destination_valid && (hit_valid || miss_valid);
  const int64_t destination_vector = plan_offset * kVectorsPerChunk + threadIdx.x;

  if (!active) {
    destination_key_values[destination_vector] = uint4{0, 0, 0, 0};
    if (threadIdx.x == 0 && row_valid && kind != -1) {
      atomicCAS(
          planner_error_codes + kv_head,
          0,
          kInvalidPlacementDescriptor);
    }
    return;
  }
  const int64_t source_vector =
      (hit_valid ? source_slot : plan_offset) * kVectorsPerChunk + threadIdx.x;
  destination_key_values[destination_vector] =
      hit_valid ? temporal_key_values[source_vector] : compatibility_key_values[source_vector];
}

__global__ void shadowkv_place_device_miss_only_kernel(
    const int8_t* __restrict__ component_kinds,
    const int32_t* __restrict__ source_slots,
    const int32_t* __restrict__ destination_slots,
    const int32_t* __restrict__ miss_ordinals,
    const int32_t* __restrict__ selected_chunk_ids,
    const int32_t* __restrict__ plan_slots,
    int32_t* __restrict__ planner_error_codes,
    const uint4* __restrict__ temporal_key_values,
    const uint4* __restrict__ reconstructed_keys,
    const int32_t* __restrict__ value_miss_chunk_ids,
    const int32_t* __restrict__ value_miss_lengths,
    const int64_t* __restrict__ descriptor_generation,
    const uint8_t* __restrict__ descriptor_validity,
    int64_t expected_generation,
    int kv_heads,
    int selected_capacity,
    int64_t temporal_source_slots,
    int plan_capacity,
    const uint4* __restrict__ value_miss_key_values,
    uint4* __restrict__ destination_key_values) {
  const int entry = blockIdx.x;
  const int selected = entry % selected_capacity;
  const int head_component = entry / selected_capacity;
  const int kv_head = head_component % kv_heads;
  const int component = head_component / kv_heads;
  const int64_t plan_offset =
      (static_cast<int64_t>(component) * kv_heads + kv_head) * selected_capacity + selected;
  const int8_t kind = component_kinds[plan_offset];
  const int plan_slot = plan_slots[kv_head];
  const int64_t expected_destination =
      ((static_cast<int64_t>(component) * plan_capacity + plan_slot) * kv_heads + kv_head) *
          selected_capacity +
      selected;
  const int64_t source_slot = source_slots[plan_offset];
  const int miss_ordinal = miss_ordinals[plan_offset];
  const bool row_valid = planner_error_codes[kv_head] == 0 && plan_slot >= 0 && plan_slot < plan_capacity;
  const bool descriptor_ready =
      descriptor_validity[0] == 1 && descriptor_generation[0] == expected_generation;
  const bool hit_valid = kind == kHit && source_slot >= 0 && source_slot < temporal_source_slots;
  const bool key_miss_valid = component == 0 && kind == kMiss && source_slot == -1 && miss_ordinal >= 0;
  const bool value_miss_valid =
      component == 1 && kind == kMiss && source_slot == -1 && miss_ordinal >= 0 &&
      miss_ordinal < value_miss_lengths[kv_head] && miss_ordinal < selected_capacity &&
      value_miss_chunk_ids[static_cast<int64_t>(kv_head) * selected_capacity + miss_ordinal] ==
          selected_chunk_ids[static_cast<int64_t>(kv_head) * selected_capacity + selected];
  const bool destination_valid = destination_slots[plan_offset] == expected_destination;
  const bool active =
      row_valid && descriptor_ready && destination_valid && (hit_valid || key_miss_valid || value_miss_valid);
  const int64_t destination_vector = plan_offset * kVectorsPerChunk + threadIdx.x;

  if (!active) {
    destination_key_values[destination_vector] = uint4{0, 0, 0, 0};
    if (threadIdx.x == 0 && row_valid && kind != -1) {
      atomicCAS(planner_error_codes + kv_head, 0, kInvalidPlacementDescriptor);
    }
    return;
  }
  int64_t source_vector = 0;
  if (hit_valid) {
    source_vector = source_slot * kVectorsPerChunk + threadIdx.x;
    destination_key_values[destination_vector] = temporal_key_values[source_vector];
  } else if (component == 0) {
    source_vector =
        (static_cast<int64_t>(kv_head) * selected_capacity + selected) * kVectorsPerChunk + threadIdx.x;
    destination_key_values[destination_vector] = reconstructed_keys[source_vector];
  } else {
    source_vector =
        (static_cast<int64_t>(kv_head) * selected_capacity + miss_ordinal) * kVectorsPerChunk + threadIdx.x;
    destination_key_values[destination_vector] = value_miss_key_values[source_vector];
  }
}

__global__ void shadowkv_validate_mapped_host_plan_kernel(
    const int8_t* __restrict__ component_kinds,
    const int32_t* __restrict__ source_slots,
    const int32_t* __restrict__ destination_slots,
    const int32_t* __restrict__ miss_ordinals,
    const int32_t* __restrict__ selected_chunk_ids,
    const int32_t* __restrict__ plan_slots,
    int32_t* __restrict__ planner_error_codes,
    const int32_t* __restrict__ value_miss_chunk_ids,
    const int32_t* __restrict__ value_miss_lengths,
    const int64_t* __restrict__ descriptor_generation,
    const uint8_t* __restrict__ descriptor_validity,
    int64_t expected_generation,
    int kv_heads,
    int selected_capacity,
    int64_t temporal_source_slots,
    int plan_capacity,
    int prompt_chunk_capacity,
    int prompt_tokens) {
  const int kv_head = blockIdx.x;
  const int plan_slot = plan_slots[kv_head];
  const int value_miss_length = value_miss_lengths[kv_head];
  if (
      descriptor_validity[0] != 1 || descriptor_generation[0] != expected_generation || plan_slot < 0 ||
      plan_slot >= plan_capacity || value_miss_length < 0 || value_miss_length > selected_capacity) {
    if (threadIdx.x == 0) {
      atomicCAS(planner_error_codes + kv_head, 0, kInvalidPlacementDescriptor);
    }
    return;
  }
  for (int selected = threadIdx.x; selected < selected_capacity; selected += blockDim.x) {
    for (int component = 0; component < kComponents; ++component) {
      const int64_t plan_offset =
          (static_cast<int64_t>(component) * kv_heads + kv_head) * selected_capacity + selected;
      const int8_t kind = component_kinds[plan_offset];
      if (kind == -1) {
        continue;
      }
      const int64_t expected_destination =
          ((static_cast<int64_t>(component) * plan_capacity + plan_slot) * kv_heads + kv_head) *
              selected_capacity +
          selected;
      const int source_slot = source_slots[plan_offset];
      const int miss_ordinal = miss_ordinals[plan_offset];
      const bool destination_valid = destination_slots[plan_offset] == expected_destination;
      const bool hit_valid = kind == kHit && source_slot >= 0 && source_slot < temporal_source_slots;
      const bool key_miss_valid =
          component == 0 && kind == kMiss && source_slot == -1 && miss_ordinal >= 0 &&
          miss_ordinal < selected_capacity;
      bool value_miss_valid = false;
      if (
          component == 1 && kind == kMiss && source_slot == -1 && miss_ordinal >= 0 &&
          miss_ordinal < value_miss_length && miss_ordinal < selected_capacity) {
        const int descriptor_chunk =
            value_miss_chunk_ids[static_cast<int64_t>(kv_head) * selected_capacity + miss_ordinal];
        const int selected_chunk =
            selected_chunk_ids[static_cast<int64_t>(kv_head) * selected_capacity + selected];
        value_miss_valid =
            descriptor_chunk == selected_chunk && descriptor_chunk >= 0 && descriptor_chunk < prompt_chunk_capacity &&
            (static_cast<int64_t>(descriptor_chunk) + 1) * kChunkSize <= prompt_tokens;
      }
      if (!destination_valid || !(hit_valid || key_miss_valid || value_miss_valid)) {
        atomicCAS(planner_error_codes + kv_head, 0, kInvalidPlacementDescriptor);
      }
    }
  }
}

__global__ void shadowkv_place_device_mapped_host_kernel(
    const int8_t* __restrict__ component_kinds,
    const int32_t* __restrict__ source_slots,
    const int32_t* __restrict__ destination_slots,
    const int32_t* __restrict__ miss_ordinals,
    const int32_t* __restrict__ selected_chunk_ids,
    const int32_t* __restrict__ plan_slots,
    int32_t* __restrict__ planner_error_codes,
    const uint4* __restrict__ temporal_key_values,
    const uint4* __restrict__ reconstructed_keys,
    const int32_t* __restrict__ value_miss_chunk_ids,
    const int32_t* __restrict__ value_miss_lengths,
    const int64_t* __restrict__ descriptor_generation,
    const uint8_t* __restrict__ descriptor_validity,
    int64_t expected_generation,
    int kv_heads,
    int selected_capacity,
    int64_t temporal_source_slots,
    int plan_capacity,
    const uint4* __restrict__ mapped_host_values,
    int prompt_chunk_capacity,
    int prompt_tokens,
    uint4* __restrict__ destination_key_values) {
  const int entry = blockIdx.x;
  const int selected = entry % selected_capacity;
  const int head_component = entry / selected_capacity;
  const int kv_head = head_component % kv_heads;
  const int component = head_component / kv_heads;
  const int64_t plan_offset =
      (static_cast<int64_t>(component) * kv_heads + kv_head) * selected_capacity + selected;
  const int8_t kind = component_kinds[plan_offset];
  const int plan_slot = plan_slots[kv_head];
  const int64_t expected_destination =
      ((static_cast<int64_t>(component) * plan_capacity + plan_slot) * kv_heads + kv_head) * selected_capacity +
      selected;
  const int64_t source_slot = source_slots[plan_offset];
  const int miss_ordinal = miss_ordinals[plan_offset];
  const bool row_valid = planner_error_codes[kv_head] == 0 && plan_slot >= 0 && plan_slot < plan_capacity;
  const bool descriptor_ready =
      descriptor_validity[0] == 1 && descriptor_generation[0] == expected_generation;
  const bool hit_valid = kind == kHit && source_slot >= 0 && source_slot < temporal_source_slots;
  const bool key_miss_valid =
      component == 0 && kind == kMiss && source_slot == -1 && miss_ordinal >= 0 &&
      miss_ordinal < selected_capacity;
  int value_chunk = -1;
  bool value_miss_valid = false;
  if (
      component == 1 && kind == kMiss && source_slot == -1 && miss_ordinal >= 0 &&
      miss_ordinal < value_miss_lengths[kv_head] && miss_ordinal < selected_capacity) {
    value_chunk = value_miss_chunk_ids[static_cast<int64_t>(kv_head) * selected_capacity + miss_ordinal];
    value_miss_valid =
        value_chunk == selected_chunk_ids[static_cast<int64_t>(kv_head) * selected_capacity + selected] &&
        value_chunk >= 0 && value_chunk < prompt_chunk_capacity &&
        (static_cast<int64_t>(value_chunk) + 1) * kChunkSize <= prompt_tokens;
  }
  const bool destination_valid = destination_slots[plan_offset] == expected_destination;
  const bool active =
      row_valid && descriptor_ready && destination_valid && (hit_valid || key_miss_valid || value_miss_valid);
  const int64_t destination_vector = plan_offset * kVectorsPerChunk + threadIdx.x;

  if (!active) {
    destination_key_values[destination_vector] = uint4{0, 0, 0, 0};
    if (threadIdx.x == 0 && row_valid && kind != -1) {
      atomicCAS(planner_error_codes + kv_head, 0, kInvalidPlacementDescriptor);
    }
    return;
  }
  int64_t source_vector = 0;
  if (hit_valid) {
    source_vector = source_slot * kVectorsPerChunk + threadIdx.x;
    destination_key_values[destination_vector] = temporal_key_values[source_vector];
  } else if (component == 0) {
    source_vector =
        (static_cast<int64_t>(kv_head) * selected_capacity + selected) * kVectorsPerChunk + threadIdx.x;
    destination_key_values[destination_vector] = reconstructed_keys[source_vector];
  } else {
    source_vector =
        (static_cast<int64_t>(kv_head) * prompt_chunk_capacity + value_chunk) * kVectorsPerChunk + threadIdx.x;
    destination_key_values[destination_vector] = mapped_host_values[source_vector];
  }
}

}  // namespace

int64_t shadowkv_resolve_mapped_host_pointer(const at::Tensor& host_values, int64_t device_index) {
  TORCH_CHECK(host_values.device().is_cpu(), "mapped host values must reside on CPU");
  TORCH_CHECK(host_values.is_contiguous(), "mapped host values must be contiguous");
  TORCH_CHECK(host_values.scalar_type() == at::ScalarType::BFloat16, "mapped host values must use bfloat16");
  TORCH_CHECK(host_values.dim() == 4, "mapped host values must have shape [heads, chunks, 8, 128]");
  TORCH_CHECK(
      host_values.size(0) >= 1 && host_values.size(1) >= 1 && host_values.size(2) == kChunkSize &&
          host_values.size(3) == kHeadDimension,
      "mapped host values must have shape [heads, chunks, 8, 128]");
  TORCH_CHECK(host_values.is_pinned(), "mapped host values must use page-locked memory");
  TORCH_CHECK(device_index >= 0 && device_index <= std::numeric_limits<int>::max(), "mapped CUDA device is invalid");
  int device_count = 0;
  C10_CUDA_CHECK(cudaGetDeviceCount(&device_count));
  TORCH_CHECK(device_index < device_count, "mapped CUDA device is not visible");
  c10::cuda::CUDAGuard device_guard(c10::Device(c10::DeviceType::CUDA, device_index));
  check_mapped_host_device(static_cast<int>(device_index));
  void* device_pointer = nullptr;
  C10_CUDA_CHECK(cudaHostGetDevicePointer(&device_pointer, host_values.data_ptr(), 0));
  TORCH_CHECK(device_pointer != nullptr, "mapped host values have no CUDA device pointer");
  cudaPointerAttributes attributes{};
  C10_CUDA_CHECK(cudaPointerGetAttributes(&attributes, host_values.data_ptr()));
  TORCH_CHECK(attributes.type == cudaMemoryTypeHost, "mapped host pointer is not registered host memory");
  TORCH_CHECK(attributes.devicePointer != nullptr, "registered host memory has no mapped device pointer");
  const uintptr_t pointer = reinterpret_cast<uintptr_t>(device_pointer);
  TORCH_CHECK(pointer % kVectorBytes == 0, "mapped host device pointer is not 16-byte aligned");
  TORCH_CHECK(
      pointer <= static_cast<uintptr_t>(std::numeric_limits<int64_t>::max()),
      "mapped host device pointer exceeds the signed operator contract");
  return static_cast<int64_t>(pointer);
}

void shadowkv_place_device(
    const at::Tensor& component_kinds,
    const at::Tensor& source_slots,
    const at::Tensor& destination_slots,
    const at::Tensor& plan_slots,
    at::Tensor& planner_error_codes,
    const at::Tensor& temporal_key_values,
    const at::Tensor& compatibility_key_values,
    int64_t plan_capacity,
    at::Tensor& destination_key_values) {
  check_tensor(component_kinds, "component_kinds", at::ScalarType::Char, 3);
  check_tensor(source_slots, "source_slots", at::ScalarType::Int, 3);
  check_tensor(destination_slots, "destination_slots", at::ScalarType::Int, 3);
  check_tensor(plan_slots, "plan_slots", at::ScalarType::Int, 1);
  check_tensor(planner_error_codes, "planner_error_codes", at::ScalarType::Int, 1);
  check_tensor(temporal_key_values, "temporal_key_values", at::ScalarType::BFloat16, 7);
  check_tensor(compatibility_key_values, "compatibility_key_values", at::ScalarType::BFloat16, 5);
  check_tensor(destination_key_values, "destination_key_values", at::ScalarType::BFloat16, 5);

  const int64_t kv_heads = component_kinds.size(1);
  const int64_t selected_capacity = component_kinds.size(2);
  const int64_t request_slots = temporal_key_values.size(1);
  const int64_t local_layers = temporal_key_values.size(2);
  const int64_t temporal_capacity = temporal_key_values.size(4);
  TORCH_CHECK(component_kinds.size(0) == kComponents, "component_kinds must have shape [2, heads, selected]");
  TORCH_CHECK(kv_heads >= 1 && selected_capacity >= 1, "placement heads and selected capacity must be positive");
  TORCH_CHECK(selected_capacity <= 256, "placement selected capacity must not exceed 256");
  TORCH_CHECK(
      kv_heads <= std::numeric_limits<int>::max() && selected_capacity <= std::numeric_limits<int>::max() &&
          plan_capacity <= std::numeric_limits<int>::max(),
      "placement dimensions exceed CUDA launch bounds");
  TORCH_CHECK(plan_capacity >= 1, "plan_capacity must be positive");
  TORCH_CHECK(
      source_slots.sizes() == component_kinds.sizes() && destination_slots.sizes() == component_kinds.sizes(),
      "placement plan tensors must share shape [2, heads, selected]");
  TORCH_CHECK(
      plan_slots.numel() == kv_heads && planner_error_codes.numel() == kv_heads,
      "placement row tensors must have shape [heads]");
  TORCH_CHECK(
      temporal_key_values.size(0) == kComponents && temporal_key_values.size(3) == kv_heads &&
          temporal_key_values.size(5) == kChunkSize && temporal_key_values.size(6) == kHeadDimension,
      "temporal_key_values must have shape [2, requests, layers, heads, temporal, 8, 128]");
  const std::vector<int64_t> output_shape = {
      kComponents, kv_heads, selected_capacity, kChunkSize, kHeadDimension};
  TORCH_CHECK(
      compatibility_key_values.sizes().vec() == output_shape && destination_key_values.sizes().vec() == output_shape,
      "compatibility and destination K/V must have shape [2, heads, selected, 8, 128]");
  TORCH_CHECK(
      request_slots >= 1 && local_layers >= 1 && temporal_capacity >= 0,
      "temporal placement dimensions are invalid");
  TORCH_CHECK(
      logical_slots_fit_int32({kComponents, request_slots, local_layers, kv_heads, temporal_capacity}),
      "temporal source slots exceed int32");
  TORCH_CHECK(
      logical_slots_fit_int32({kComponents, plan_capacity, kv_heads, selected_capacity}),
      "destination slots exceed int32");
  const int64_t temporal_source_slots =
      kComponents * request_slots * local_layers * kv_heads * temporal_capacity;

  const auto device = component_kinds.device();
  const at::Tensor* tensors[] = {
      &source_slots,
      &destination_slots,
      &plan_slots,
      &planner_error_codes,
      &temporal_key_values,
      &compatibility_key_values,
      &destination_key_values,
  };
  for (const at::Tensor* tensor : tensors) {
    TORCH_CHECK(tensor->device() == device, "all shadowkv_place_device tensors must share one CUDA device");
  }

  c10::cuda::CUDAGuard device_guard(device);
  check_b200(component_kinds);
  const int64_t entries = kComponents * kv_heads * selected_capacity;
  TORCH_CHECK(entries <= std::numeric_limits<int>::max(), "placement grid exceeds CUDA launch bounds");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  shadowkv_place_device_kernel<<<static_cast<int>(entries), kCopyThreads, 0, stream>>>(
      component_kinds.data_ptr<int8_t>(),
      source_slots.data_ptr<int32_t>(),
      destination_slots.data_ptr<int32_t>(),
      plan_slots.data_ptr<int32_t>(),
      planner_error_codes.data_ptr<int32_t>(),
      reinterpret_cast<const uint4*>(temporal_key_values.data_ptr<at::BFloat16>()),
      reinterpret_cast<const uint4*>(compatibility_key_values.data_ptr<at::BFloat16>()),
      static_cast<int>(kv_heads),
      static_cast<int>(selected_capacity),
      temporal_source_slots,
      static_cast<int>(plan_capacity),
      reinterpret_cast<uint4*>(destination_key_values.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void shadowkv_place_device_miss_only(
    const at::Tensor& component_kinds,
    const at::Tensor& source_slots,
    const at::Tensor& destination_slots,
    const at::Tensor& miss_ordinals,
    const at::Tensor& selected_chunk_ids,
    const at::Tensor& plan_slots,
    at::Tensor& planner_error_codes,
    const at::Tensor& temporal_key_values,
    const at::Tensor& reconstructed_keys,
    const at::Tensor& value_miss_chunk_ids,
    const at::Tensor& value_miss_lengths,
    const at::Tensor& descriptor_generation,
    const at::Tensor& descriptor_validity,
    int64_t expected_generation,
    int64_t plan_capacity,
    const at::Tensor& value_miss_key_values,
    at::Tensor& destination_key_values) {
  check_tensor(component_kinds, "component_kinds", at::ScalarType::Char, 3);
  check_tensor(source_slots, "source_slots", at::ScalarType::Int, 3);
  check_tensor(destination_slots, "destination_slots", at::ScalarType::Int, 3);
  check_tensor(miss_ordinals, "miss_ordinals", at::ScalarType::Int, 3);
  check_tensor(selected_chunk_ids, "selected_chunk_ids", at::ScalarType::Int, 2);
  check_tensor(plan_slots, "plan_slots", at::ScalarType::Int, 1);
  check_tensor(planner_error_codes, "planner_error_codes", at::ScalarType::Int, 1);
  check_tensor(temporal_key_values, "temporal_key_values", at::ScalarType::BFloat16, 7);
  check_tensor(reconstructed_keys, "reconstructed_keys", at::ScalarType::BFloat16, 4);
  check_tensor(value_miss_chunk_ids, "value_miss_chunk_ids", at::ScalarType::Int, 2);
  check_tensor(value_miss_lengths, "value_miss_lengths", at::ScalarType::Int, 1);
  check_tensor(descriptor_generation, "descriptor_generation", at::ScalarType::Long, 1);
  check_tensor(descriptor_validity, "descriptor_validity", at::ScalarType::Byte, 1);
  check_tensor(value_miss_key_values, "value_miss_key_values", at::ScalarType::BFloat16, 4);
  check_tensor(destination_key_values, "destination_key_values", at::ScalarType::BFloat16, 5);

  const int64_t kv_heads = component_kinds.size(1);
  const int64_t selected_capacity = component_kinds.size(2);
  const int64_t request_slots = temporal_key_values.size(1);
  const int64_t local_layers = temporal_key_values.size(2);
  const int64_t temporal_capacity = temporal_key_values.size(4);
  TORCH_CHECK(component_kinds.size(0) == kComponents, "component_kinds must have shape [2, heads, selected]");
  TORCH_CHECK(kv_heads >= 1 && selected_capacity >= 1, "placement heads and selected capacity must be positive");
  TORCH_CHECK(selected_capacity <= 256, "placement selected capacity must not exceed 256");
  TORCH_CHECK(
      kv_heads <= std::numeric_limits<int>::max() && selected_capacity <= std::numeric_limits<int>::max() &&
          plan_capacity <= std::numeric_limits<int>::max(),
      "placement dimensions exceed CUDA launch bounds");
  TORCH_CHECK(plan_capacity >= 1, "plan_capacity must be positive");
  TORCH_CHECK(expected_generation >= 0, "expected_generation must be nonnegative");
  TORCH_CHECK(
      source_slots.sizes() == component_kinds.sizes() && destination_slots.sizes() == component_kinds.sizes() &&
          miss_ordinals.sizes() == component_kinds.sizes(),
      "placement plan tensors must share shape [2, heads, selected]");
  TORCH_CHECK(
      selected_chunk_ids.sizes() == at::IntArrayRef({kv_heads, selected_capacity}) &&
          value_miss_chunk_ids.sizes() == selected_chunk_ids.sizes(),
      "selected and compact value-miss chunk ids must have shape [heads, selected]");
  TORCH_CHECK(
      plan_slots.numel() == kv_heads && planner_error_codes.numel() == kv_heads &&
          value_miss_lengths.numel() == kv_heads,
      "placement row tensors must have shape [heads]");
  TORCH_CHECK(
      descriptor_generation.numel() == 1 && descriptor_validity.numel() == 1,
      "descriptor generation and validity must contain one slot scalar");
  TORCH_CHECK(
      temporal_key_values.size(0) == kComponents && temporal_key_values.size(3) == kv_heads &&
          temporal_key_values.size(5) == kChunkSize && temporal_key_values.size(6) == kHeadDimension,
      "temporal_key_values must have shape [2, requests, layers, heads, temporal, 8, 128]");
  const std::vector<int64_t> compact_shape = {kv_heads, selected_capacity, kChunkSize, kHeadDimension};
  const std::vector<int64_t> output_shape = {
      kComponents, kv_heads, selected_capacity, kChunkSize, kHeadDimension};
  TORCH_CHECK(
      reconstructed_keys.sizes().vec() == compact_shape && value_miss_key_values.sizes().vec() == compact_shape,
      "reconstructed keys and compact value misses must have shape [heads, selected, 8, 128]");
  TORCH_CHECK(
      destination_key_values.sizes().vec() == output_shape,
      "destination K/V must have shape [2, heads, selected, 8, 128]");
  TORCH_CHECK(
      request_slots >= 1 && local_layers >= 1 && temporal_capacity >= 0,
      "temporal placement dimensions are invalid");
  TORCH_CHECK(
      logical_slots_fit_int32({kComponents, request_slots, local_layers, kv_heads, temporal_capacity}),
      "temporal source slots exceed int32");
  TORCH_CHECK(
      logical_slots_fit_int32({kComponents, plan_capacity, kv_heads, selected_capacity}),
      "destination slots exceed int32");
  const int64_t temporal_source_slots =
      kComponents * request_slots * local_layers * kv_heads * temporal_capacity;

  const auto device = component_kinds.device();
  const at::Tensor* tensors[] = {
      &source_slots,
      &destination_slots,
      &miss_ordinals,
      &selected_chunk_ids,
      &plan_slots,
      &planner_error_codes,
      &temporal_key_values,
      &reconstructed_keys,
      &value_miss_chunk_ids,
      &value_miss_lengths,
      &descriptor_generation,
      &descriptor_validity,
      &value_miss_key_values,
      &destination_key_values,
  };
  for (const at::Tensor* tensor : tensors) {
    TORCH_CHECK(tensor->device() == device, "all shadowkv_place_device_miss_only tensors must share one CUDA device");
  }

  c10::cuda::CUDAGuard device_guard(device);
  check_b200(component_kinds);
  const int64_t entries = kComponents * kv_heads * selected_capacity;
  TORCH_CHECK(entries <= std::numeric_limits<int>::max(), "placement grid exceeds CUDA launch bounds");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  shadowkv_place_device_miss_only_kernel<<<static_cast<int>(entries), kCopyThreads, 0, stream>>>(
      component_kinds.data_ptr<int8_t>(),
      source_slots.data_ptr<int32_t>(),
      destination_slots.data_ptr<int32_t>(),
      miss_ordinals.data_ptr<int32_t>(),
      selected_chunk_ids.data_ptr<int32_t>(),
      plan_slots.data_ptr<int32_t>(),
      planner_error_codes.data_ptr<int32_t>(),
      reinterpret_cast<const uint4*>(temporal_key_values.data_ptr<at::BFloat16>()),
      reinterpret_cast<const uint4*>(reconstructed_keys.data_ptr<at::BFloat16>()),
      value_miss_chunk_ids.data_ptr<int32_t>(),
      value_miss_lengths.data_ptr<int32_t>(),
      descriptor_generation.data_ptr<int64_t>(),
      descriptor_validity.data_ptr<uint8_t>(),
      expected_generation,
      static_cast<int>(kv_heads),
      static_cast<int>(selected_capacity),
      temporal_source_slots,
      static_cast<int>(plan_capacity),
      reinterpret_cast<const uint4*>(value_miss_key_values.data_ptr<at::BFloat16>()),
      reinterpret_cast<uint4*>(destination_key_values.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void shadowkv_place_device_mapped_host(
    const at::Tensor& component_kinds,
    const at::Tensor& source_slots,
    const at::Tensor& destination_slots,
    const at::Tensor& miss_ordinals,
    const at::Tensor& selected_chunk_ids,
    const at::Tensor& plan_slots,
    at::Tensor& planner_error_codes,
    const at::Tensor& temporal_key_values,
    const at::Tensor& reconstructed_keys,
    const at::Tensor& value_miss_chunk_ids,
    const at::Tensor& value_miss_lengths,
    const at::Tensor& descriptor_generation,
    const at::Tensor& descriptor_validity,
    int64_t mapped_host_pointer,
    int64_t mapped_host_bytes,
    int64_t prompt_chunk_capacity,
    int64_t prompt_tokens,
    int64_t expected_generation,
    int64_t plan_capacity,
    at::Tensor& destination_key_values) {
  check_tensor(component_kinds, "component_kinds", at::ScalarType::Char, 3);
  check_tensor(source_slots, "source_slots", at::ScalarType::Int, 3);
  check_tensor(destination_slots, "destination_slots", at::ScalarType::Int, 3);
  check_tensor(miss_ordinals, "miss_ordinals", at::ScalarType::Int, 3);
  check_tensor(selected_chunk_ids, "selected_chunk_ids", at::ScalarType::Int, 2);
  check_tensor(plan_slots, "plan_slots", at::ScalarType::Int, 1);
  check_tensor(planner_error_codes, "planner_error_codes", at::ScalarType::Int, 1);
  check_tensor(temporal_key_values, "temporal_key_values", at::ScalarType::BFloat16, 7);
  check_tensor(reconstructed_keys, "reconstructed_keys", at::ScalarType::BFloat16, 4);
  check_tensor(value_miss_chunk_ids, "value_miss_chunk_ids", at::ScalarType::Int, 2);
  check_tensor(value_miss_lengths, "value_miss_lengths", at::ScalarType::Int, 1);
  check_tensor(descriptor_generation, "descriptor_generation", at::ScalarType::Long, 1);
  check_tensor(descriptor_validity, "descriptor_validity", at::ScalarType::Byte, 1);
  check_tensor(destination_key_values, "destination_key_values", at::ScalarType::BFloat16, 5);

  const int64_t kv_heads = component_kinds.size(1);
  const int64_t selected_capacity = component_kinds.size(2);
  const int64_t request_slots = temporal_key_values.size(1);
  const int64_t local_layers = temporal_key_values.size(2);
  const int64_t temporal_capacity = temporal_key_values.size(4);
  TORCH_CHECK(component_kinds.size(0) == kComponents, "component_kinds must have shape [2, heads, selected]");
  TORCH_CHECK(kv_heads >= 1 && selected_capacity >= 1, "placement heads and selected capacity must be positive");
  TORCH_CHECK(selected_capacity <= 256, "placement selected capacity must not exceed 256");
  TORCH_CHECK(
      kv_heads <= std::numeric_limits<int>::max() && selected_capacity <= std::numeric_limits<int>::max() &&
          plan_capacity <= std::numeric_limits<int>::max() &&
          prompt_chunk_capacity <= std::numeric_limits<int>::max() && prompt_tokens <= std::numeric_limits<int>::max(),
      "mapped placement dimensions exceed CUDA launch bounds");
  TORCH_CHECK(plan_capacity >= 1, "plan_capacity must be positive");
  TORCH_CHECK(expected_generation >= 0, "expected_generation must be nonnegative");
  TORCH_CHECK(prompt_chunk_capacity >= 1, "prompt chunk capacity must be positive");
  TORCH_CHECK(
      prompt_tokens >= 1 && prompt_tokens <= prompt_chunk_capacity * kChunkSize,
      "prompt token bound exceeds mapped host storage");
  TORCH_CHECK(mapped_host_pointer > 0, "mapped host pointer must be positive");
  TORCH_CHECK(mapped_host_pointer % kVectorBytes == 0, "mapped host pointer must be 16-byte aligned");
  TORCH_CHECK(mapped_host_bytes > 0, "mapped host byte range must be positive");
  TORCH_CHECK(
      kv_heads <= std::numeric_limits<int64_t>::max() / prompt_chunk_capacity / kChunkBytes,
      "mapped host byte range overflows int64");
  const int64_t required_mapped_bytes = kv_heads * prompt_chunk_capacity * kChunkBytes;
  TORCH_CHECK(mapped_host_bytes >= required_mapped_bytes, "mapped host byte range is smaller than its declared shape");
  TORCH_CHECK(
      source_slots.sizes() == component_kinds.sizes() && destination_slots.sizes() == component_kinds.sizes() &&
          miss_ordinals.sizes() == component_kinds.sizes(),
      "placement plan tensors must share shape [2, heads, selected]");
  TORCH_CHECK(
      selected_chunk_ids.sizes() == at::IntArrayRef({kv_heads, selected_capacity}) &&
          value_miss_chunk_ids.sizes() == selected_chunk_ids.sizes(),
      "selected and compact value-miss chunk ids must have shape [heads, selected]");
  TORCH_CHECK(
      plan_slots.numel() == kv_heads && planner_error_codes.numel() == kv_heads &&
          value_miss_lengths.numel() == kv_heads,
      "placement row tensors must have shape [heads]");
  TORCH_CHECK(
      descriptor_generation.numel() == 1 && descriptor_validity.numel() == 1,
      "descriptor generation and validity must contain one slot scalar");
  TORCH_CHECK(
      temporal_key_values.size(0) == kComponents && temporal_key_values.size(3) == kv_heads &&
          temporal_key_values.size(5) == kChunkSize && temporal_key_values.size(6) == kHeadDimension,
      "temporal_key_values must have shape [2, requests, layers, heads, temporal, 8, 128]");
  const std::vector<int64_t> compact_shape = {kv_heads, selected_capacity, kChunkSize, kHeadDimension};
  const std::vector<int64_t> output_shape = {
      kComponents, kv_heads, selected_capacity, kChunkSize, kHeadDimension};
  TORCH_CHECK(
      reconstructed_keys.sizes().vec() == compact_shape,
      "reconstructed keys must have shape [heads, selected, 8, 128]");
  TORCH_CHECK(
      destination_key_values.sizes().vec() == output_shape,
      "destination K/V must have shape [2, heads, selected, 8, 128]");
  TORCH_CHECK(
      request_slots >= 1 && local_layers >= 1 && temporal_capacity >= 0,
      "temporal placement dimensions are invalid");
  TORCH_CHECK(
      logical_slots_fit_int32({kComponents, request_slots, local_layers, kv_heads, temporal_capacity}),
      "temporal source slots exceed int32");
  TORCH_CHECK(
      logical_slots_fit_int32({kComponents, plan_capacity, kv_heads, selected_capacity}),
      "destination slots exceed int32");
  const int64_t temporal_source_slots =
      kComponents * request_slots * local_layers * kv_heads * temporal_capacity;

  const auto device = component_kinds.device();
  const at::Tensor* tensors[] = {
      &source_slots,
      &destination_slots,
      &miss_ordinals,
      &selected_chunk_ids,
      &plan_slots,
      &planner_error_codes,
      &temporal_key_values,
      &reconstructed_keys,
      &value_miss_chunk_ids,
      &value_miss_lengths,
      &descriptor_generation,
      &descriptor_validity,
      &destination_key_values,
  };
  for (const at::Tensor* tensor : tensors) {
    TORCH_CHECK(tensor->device() == device, "all shadowkv_place_device_mapped_host tensors must share one CUDA device");
  }

  c10::cuda::CUDAGuard device_guard(device);
  check_b200(component_kinds);
  check_mapped_host_device(component_kinds.get_device());
  const int64_t entries = kComponents * kv_heads * selected_capacity;
  TORCH_CHECK(entries <= std::numeric_limits<int>::max(), "placement grid exceeds CUDA launch bounds");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  shadowkv_validate_mapped_host_plan_kernel<<<static_cast<int>(kv_heads), 256, 0, stream>>>(
      component_kinds.data_ptr<int8_t>(),
      source_slots.data_ptr<int32_t>(),
      destination_slots.data_ptr<int32_t>(),
      miss_ordinals.data_ptr<int32_t>(),
      selected_chunk_ids.data_ptr<int32_t>(),
      plan_slots.data_ptr<int32_t>(),
      planner_error_codes.data_ptr<int32_t>(),
      value_miss_chunk_ids.data_ptr<int32_t>(),
      value_miss_lengths.data_ptr<int32_t>(),
      descriptor_generation.data_ptr<int64_t>(),
      descriptor_validity.data_ptr<uint8_t>(),
      expected_generation,
      static_cast<int>(kv_heads),
      static_cast<int>(selected_capacity),
      temporal_source_slots,
      static_cast<int>(plan_capacity),
      static_cast<int>(prompt_chunk_capacity),
      static_cast<int>(prompt_tokens));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  shadowkv_place_device_mapped_host_kernel<<<static_cast<int>(entries), kCopyThreads, 0, stream>>>(
      component_kinds.data_ptr<int8_t>(),
      source_slots.data_ptr<int32_t>(),
      destination_slots.data_ptr<int32_t>(),
      miss_ordinals.data_ptr<int32_t>(),
      selected_chunk_ids.data_ptr<int32_t>(),
      plan_slots.data_ptr<int32_t>(),
      planner_error_codes.data_ptr<int32_t>(),
      reinterpret_cast<const uint4*>(temporal_key_values.data_ptr<at::BFloat16>()),
      reinterpret_cast<const uint4*>(reconstructed_keys.data_ptr<at::BFloat16>()),
      value_miss_chunk_ids.data_ptr<int32_t>(),
      value_miss_lengths.data_ptr<int32_t>(),
      descriptor_generation.data_ptr<int64_t>(),
      descriptor_validity.data_ptr<uint8_t>(),
      expected_generation,
      static_cast<int>(kv_heads),
      static_cast<int>(selected_capacity),
      temporal_source_slots,
      static_cast<int>(plan_capacity),
      reinterpret_cast<const uint4*>(static_cast<uintptr_t>(mapped_host_pointer)),
      static_cast<int>(prompt_chunk_capacity),
      static_cast<int>(prompt_tokens),
      reinterpret_cast<uint4*>(destination_key_values.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
