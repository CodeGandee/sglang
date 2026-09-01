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
#include <mma.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>
#include <limits>

#include "utils.h"

namespace {

constexpr int8_t kInactive = -1;
constexpr int8_t kHit = 1;
constexpr int8_t kMiss = 2;
constexpr int32_t kInvalidPlacementDescriptor = 8;
constexpr int kComponents = 2;
constexpr int kKVHeads = 8;
constexpr int kSelectedCapacity = 256;
constexpr int kChunkSize = 8;
constexpr int kRank = 160;
constexpr int kHeadDimension = 128;
constexpr int kHalfHeadDimension = kHeadDimension / 2;
constexpr int kWmmaTile = 16;
constexpr int kChunksPerKeyBlock = 16;
constexpr int kRowsPerKeyBlock = kChunksPerKeyBlock * kChunkSize;
constexpr int kKeyBlocksPerHead = kSelectedCapacity / kChunksPerKeyBlock;
constexpr int kWarpRows = 64;
constexpr int kWarpColumns = 32;
constexpr int kWarpRowTiles = kWarpRows / kWmmaTile;
constexpr int kWarpColumnTiles = kWarpColumns / kWmmaTile;
constexpr int kKeyBlockColumnTiles = kHeadDimension / kWarpColumns;
constexpr int kKeyBlockRowTiles = kRowsPerKeyBlock / kWarpRows;
constexpr int kThreads =
    kKeyBlockColumnTiles * kKeyBlockRowTiles * 32;
constexpr int kChunkElements = kChunkSize * kHeadDimension;
constexpr int kVectorBytes = sizeof(uint4);
constexpr int kChunkBytes = kChunkElements * sizeof(at::BFloat16);
constexpr int kVectorsPerChunk = kChunkBytes / kVectorBytes;

static_assert(kVectorsPerChunk <= kThreads);
static_assert(kThreads <= 1024 && kThreads % 32 == 0);
static_assert(kSelectedCapacity % kChunksPerKeyBlock == 0);
static_assert(kRowsPerKeyBlock == 128 && kThreads == 256);

void check_tensor(
    const at::Tensor& tensor,
    const char* name,
    at::ScalarType dtype,
    int64_t dimensions) {
  CHECK_INPUT(tensor);
  TORCH_CHECK(
      tensor.dim() == dimensions,
      name,
      " must have ",
      dimensions,
      " dimensions");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an invalid dtype");
}

void check_sm80(const at::Tensor& tensor) {
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, tensor.get_device()));
  TORCH_CHECK(
      properties.major == 8 && properties.minor == 0,
      "A100 fused ShadowKV kernels require compute capability 8.0; found ",
      properties.major,
      ".",
      properties.minor);
}

struct PlanGeometry {
  int64_t request_slots;
  int64_t local_layers;
  int64_t temporal_capacity;
  int64_t temporal_chunks_per_component;
};

PlanGeometry check_plan_tensors(
    const at::Tensor& component_kinds,
    const at::Tensor& source_slots,
    const at::Tensor& destination_slots,
    const at::Tensor& selected_lengths,
    const at::Tensor& plan_slots,
    const at::Tensor& planner_error_codes,
    const at::Tensor& temporal_key_values,
    int64_t plan_capacity,
    const at::Tensor& destination_key_values) {
  check_tensor(component_kinds, "component_kinds", at::ScalarType::Char, 3);
  check_tensor(source_slots, "source_slots", at::ScalarType::Int, 3);
  check_tensor(destination_slots, "destination_slots", at::ScalarType::Int, 3);
  check_tensor(selected_lengths, "selected_lengths", at::ScalarType::Int, 1);
  check_tensor(plan_slots, "plan_slots", at::ScalarType::Int, 1);
  check_tensor(planner_error_codes, "planner_error_codes", at::ScalarType::Int, 1);
  check_tensor(temporal_key_values, "temporal_key_values", at::ScalarType::BFloat16, 7);
  check_tensor(destination_key_values, "destination_key_values", at::ScalarType::BFloat16, 5);

  const auto plan_shape = at::IntArrayRef({kComponents, kKVHeads, kSelectedCapacity});
  TORCH_CHECK(component_kinds.sizes() == plan_shape, "component_kinds must have shape [2, 8, 256]");
  TORCH_CHECK(
      source_slots.sizes() == plan_shape && destination_slots.sizes() == plan_shape,
      "source and destination slots must have shape [2, 8, 256]");
  TORCH_CHECK(
      selected_lengths.sizes() == at::IntArrayRef({kKVHeads}) &&
          plan_slots.sizes() == at::IntArrayRef({kKVHeads}) &&
          planner_error_codes.sizes() == at::IntArrayRef({kKVHeads}),
      "plan rows must have shape [8]");
  TORCH_CHECK(
      temporal_key_values.size(0) == kComponents &&
          temporal_key_values.size(3) == kKVHeads &&
          temporal_key_values.size(5) == kChunkSize &&
          temporal_key_values.size(6) == kHeadDimension,
      "temporal_key_values must have shape [2, requests, layers, 8, temporal, 8, 128]");
  TORCH_CHECK(
      destination_key_values.sizes() ==
          at::IntArrayRef({kComponents, kKVHeads, kSelectedCapacity, kChunkSize, kHeadDimension}),
      "destination_key_values must have shape [2, 8, 256, 8, 128]");
  TORCH_CHECK(
      plan_capacity >= 1 && plan_capacity <= std::numeric_limits<int>::max(),
      "plan_capacity is outside the A100 fused launch envelope");
  const int64_t request_slots = temporal_key_values.size(1);
  const int64_t local_layers = temporal_key_values.size(2);
  const int64_t temporal_capacity = temporal_key_values.size(4);
  TORCH_CHECK(
      request_slots >= 1 && local_layers >= 1 && temporal_capacity >= 0,
      "temporal storage dimensions are invalid");
  TORCH_CHECK(
      request_slots <= std::numeric_limits<int32_t>::max() / local_layers / kKVHeads /
              std::max<int64_t>(temporal_capacity, 1),
      "temporal source slots exceed int32");
  TORCH_CHECK(
      plan_capacity <= std::numeric_limits<int32_t>::max() / kComponents / kKVHeads /
              kSelectedCapacity,
      "destination slots exceed int32");

  const auto device = component_kinds.device();
  const at::Tensor* tensors[] = {
      &source_slots,
      &destination_slots,
      &selected_lengths,
      &plan_slots,
      &planner_error_codes,
      &temporal_key_values,
      &destination_key_values,
  };
  for (const at::Tensor* tensor : tensors) {
    TORCH_CHECK(
        tensor->device() == device,
        "all A100 fused plan tensors must share one CUDA device");
  }
  return PlanGeometry{
      request_slots,
      local_layers,
      temporal_capacity,
      request_slots * local_layers * kKVHeads * temporal_capacity};
}

__device__ __forceinline__ void mark_invalid(
    int32_t* planner_error_codes,
    int head,
    bool row_valid) {
  if (threadIdx.x == 0 && row_valid) {
    atomicCAS(
        planner_error_codes + head,
        0,
        kInvalidPlacementDescriptor);
  }
}

__global__ void shadowkv_fused_key_a100_kernel(
    const __nv_bfloat16* __restrict__ u,
    const __nv_bfloat16* __restrict__ sv,
    const float* __restrict__ cosine,
    const float* __restrict__ sine,
    const int8_t* __restrict__ component_kinds,
    const int32_t* __restrict__ source_slots,
    const int32_t* __restrict__ destination_slots,
    const int32_t* __restrict__ miss_ordinals,
    const int32_t* __restrict__ selected_chunk_ids,
    const int32_t* __restrict__ selected_lengths,
    const int32_t* __restrict__ plan_slots,
    int32_t* __restrict__ planner_error_codes,
    const uint4* __restrict__ temporal_key_values,
    int u_tokens,
    int rope_rows,
    int64_t temporal_chunks_per_component,
    int plan_capacity,
    uint4* __restrict__ destination_key_values) {
  const int key_block = blockIdx.x % kKeyBlocksPerHead;
  const int head = blockIdx.x / kKeyBlocksPerHead;
  const int selected_base = key_block * kChunksPerKeyBlock;
  const int selected_length = selected_lengths[head];
  const int plan_slot = plan_slots[head];
  const bool plan_row_valid =
      planner_error_codes[head] == 0 && plan_slot >= 0 &&
      plan_slot < plan_capacity && selected_length >= 0 &&
      selected_length <= kSelectedCapacity;
  __shared__ int8_t kinds[kChunksPerKeyBlock];
  __shared__ int32_t temporal_sources[kChunksPerKeyBlock];
  __shared__ int32_t chunks[kChunksPerKeyBlock];
  __shared__ int has_miss;
  __shared__ __align__(32) __nv_bfloat16
      selected_u[kRowsPerKeyBlock][kRank];

  if (threadIdx.x == 0) {
    has_miss = 0;
  }
  if (threadIdx.x < kChunksPerKeyBlock) {
    const int local_chunk = threadIdx.x;
    const int selected = selected_base + local_chunk;
    const int64_t plan_offset =
        static_cast<int64_t>(head) * kSelectedCapacity + selected;
    const int8_t kind = component_kinds[plan_offset];
    const int source_slot = source_slots[plan_offset];
    const int destination_slot = destination_slots[plan_offset];
    const int miss_ordinal = miss_ordinals[plan_offset];
    const int selected_chunk = selected_chunk_ids[plan_offset];
    const bool active_ordinal = selected < selected_length;
    const int64_t expected_destination =
        (static_cast<int64_t>(plan_slot) * kKVHeads + head) *
            kSelectedCapacity +
        selected;
    const bool hit_valid =
        active_ordinal && kind == kHit && source_slot >= 0 &&
        source_slot < temporal_chunks_per_component && miss_ordinal == -1;
    const bool miss_valid =
        active_ordinal && kind == kMiss && source_slot == -1 &&
        miss_ordinal >= 0 && miss_ordinal < kSelectedCapacity &&
        selected_chunk >= 0 &&
        (static_cast<int64_t>(selected_chunk) + 1) * kChunkSize <=
            u_tokens &&
        (static_cast<int64_t>(selected_chunk) + 1) * kChunkSize <=
            rope_rows;
    const bool inactive_valid =
        !active_ordinal && kind == kInactive && source_slot == -1 &&
        destination_slot == -1 && miss_ordinal == -1;
    const bool destination_valid =
        active_ordinal && destination_slot == expected_destination;
    const bool entry_valid =
        plan_row_valid &&
        (inactive_valid || (destination_valid && (hit_valid || miss_valid)));
    kinds[local_chunk] = entry_valid ? kind : kInactive;
    temporal_sources[local_chunk] = entry_valid ? source_slot : -1;
    chunks[local_chunk] = entry_valid ? selected_chunk : -1;
    if (entry_valid && miss_valid) {
      atomicExch(&has_miss, 1);
    }
    if (plan_row_valid && !entry_valid) {
      atomicCAS(
          planner_error_codes + head,
          0,
          kInvalidPlacementDescriptor);
    }
  }
  __syncthreads();

  __nv_bfloat16* destination_elements =
      reinterpret_cast<__nv_bfloat16*>(destination_key_values);
  const int64_t destination_block_element =
      (static_cast<int64_t>(head) * kSelectedCapacity + selected_base) *
      kChunkElements;
  if (has_miss) {
    for (int index = threadIdx.x;
         index < kRowsPerKeyBlock * kRank;
         index += kThreads) {
      const int row = index / kRank;
      const int rank = index - row * kRank;
      const int local_chunk = row / kChunkSize;
      const int token = row - local_chunk * kChunkSize;
      const int selected_chunk = chunks[local_chunk];
      selected_u[row][rank] = kinds[local_chunk] == kMiss
          ? u[(static_cast<int64_t>(selected_chunk) * kChunkSize + token) *
                  kRank +
              rank]
          : __float2bfloat16_rn(0.0f);
    }
    __syncthreads();

    using namespace nvcuda;
    const int warp = threadIdx.x / 32;
    const int warp_row = warp / kKeyBlockColumnTiles;
    const int warp_column = warp - warp_row * kKeyBlockColumnTiles;
    wmma::fragment<
        wmma::matrix_a,
        kWmmaTile,
        kWmmaTile,
        kWmmaTile,
        __nv_bfloat16,
        wmma::row_major>
        a_fragments[kWarpRowTiles];
    wmma::fragment<
        wmma::matrix_b,
        kWmmaTile,
        kWmmaTile,
        kWmmaTile,
        __nv_bfloat16,
        wmma::row_major>
        b_fragments[kWarpColumnTiles];
    wmma::fragment<
        wmma::accumulator,
        kWmmaTile,
        kWmmaTile,
        kWmmaTile,
        float>
        accumulators[kWarpRowTiles][kWarpColumnTiles];
#pragma unroll
    for (int row_tile = 0; row_tile < kWarpRowTiles; ++row_tile) {
#pragma unroll
      for (int column_tile = 0; column_tile < kWarpColumnTiles;
           ++column_tile) {
        wmma::fill_fragment(
            accumulators[row_tile][column_tile],
            0.0f);
      }
    }
#pragma unroll
    for (int rank = 0; rank < kRank; rank += kWmmaTile) {
#pragma unroll
      for (int row_tile = 0; row_tile < kWarpRowTiles; ++row_tile) {
        wmma::load_matrix_sync(
            a_fragments[row_tile],
            &selected_u[warp_row * kWarpRows + row_tile * kWmmaTile][rank],
            kRank);
      }
#pragma unroll
      for (int column_tile = 0; column_tile < kWarpColumnTiles;
           ++column_tile) {
        wmma::load_matrix_sync(
            b_fragments[column_tile],
            sv +
                (static_cast<int64_t>(head) * kRank + rank) *
                    kHeadDimension +
                warp_column * kWarpColumns + column_tile * kWmmaTile,
            kHeadDimension);
      }
#pragma unroll
      for (int row_tile = 0; row_tile < kWarpRowTiles; ++row_tile) {
#pragma unroll
        for (int column_tile = 0; column_tile < kWarpColumnTiles;
             ++column_tile) {
          wmma::mma_sync(
              accumulators[row_tile][column_tile],
              a_fragments[row_tile],
              b_fragments[column_tile],
              accumulators[row_tile][column_tile]);
        }
      }
    }
    __syncthreads();
    float* warp_scratch =
        reinterpret_cast<float*>(selected_u) + warp * kWmmaTile * kWmmaTile;
    const int lane = threadIdx.x % 32;
#pragma unroll
    for (int row_tile = 0; row_tile < kWarpRowTiles; ++row_tile) {
#pragma unroll
      for (int column_tile = 0; column_tile < kWarpColumnTiles;
           ++column_tile) {
        wmma::store_matrix_sync(
            warp_scratch,
            accumulators[row_tile][column_tile],
            kWmmaTile,
            wmma::mem_row_major);
        __syncwarp();
        for (int tile_element = lane;
             tile_element < kWmmaTile * kWmmaTile;
             tile_element += 32) {
          const int tile_row = tile_element / kWmmaTile;
          const int tile_column = tile_element - tile_row * kWmmaTile;
          const int output_row =
              warp_row * kWarpRows + row_tile * kWmmaTile + tile_row;
          const int output_column =
              warp_column * kWarpColumns + column_tile * kWmmaTile +
              tile_column;
          destination_elements[
              destination_block_element + output_row * kHeadDimension +
              output_column] =
                  __float2bfloat16_rn(warp_scratch[tile_element]);
        }
        __syncwarp();
      }
    }
    __syncthreads();
  }

  if (has_miss) {
    for (int index = threadIdx.x;
         index < kRowsPerKeyBlock * kHeadDimension;
         index += kThreads) {
      const int row = index / kHeadDimension;
      const int dimension = index - row * kHeadDimension;
      selected_u[row][dimension] =
          destination_elements[destination_block_element + index];
    }
    __syncthreads();
  }

  for (int index = threadIdx.x;
       index < kRowsPerKeyBlock * kHeadDimension;
       index += kThreads) {
    const int row = index / kHeadDimension;
    const int dimension = index - row * kHeadDimension;
    const int local_chunk = row / kChunkSize;
    const int token = row - local_chunk * kChunkSize;
    const int8_t kind = kinds[local_chunk];
    const int64_t destination_element = destination_block_element + index;
    if (kind == kMiss) {
      const int paired_dimension =
          dimension < kHalfHeadDimension
          ? dimension + kHalfHeadDimension
          : dimension - kHalfHeadDimension;
      const int frequency = dimension % kHalfHeadDimension;
      const int64_t position =
          static_cast<int64_t>(chunks[local_chunk]) * kChunkSize + token;
      const float value =
          __bfloat162float(selected_u[row][dimension]);
      const float paired = __bfloat162float(
          selected_u[row][paired_dimension]);
      const float rotated_half =
          dimension < kHalfHeadDimension ? -paired : paired;
      const int64_t frequency_offset =
          position * kHalfHeadDimension + frequency;
      const float direct_product =
          __fmul_rn(value, cosine[frequency_offset]);
      const float rotated_product =
          __fmul_rn(rotated_half, sine[frequency_offset]);
      destination_elements[destination_element] =
          __float2bfloat16_rn(
              __fadd_rn(direct_product, rotated_product));
    } else if (kind == kInactive) {
      destination_elements[destination_element] =
          __float2bfloat16_rn(0.0f);
    }
  }
  __syncthreads();

  for (int index = threadIdx.x;
       index < kChunksPerKeyBlock * kVectorsPerChunk;
       index += kThreads) {
    const int local_chunk = index / kVectorsPerChunk;
    const int vector = index - local_chunk * kVectorsPerChunk;
    if (kinds[local_chunk] != kHit) {
      continue;
    }
    const int64_t destination_vector =
        (static_cast<int64_t>(head) * kSelectedCapacity + selected_base +
         local_chunk) *
            kVectorsPerChunk +
        vector;
    destination_key_values[destination_vector] =
        temporal_key_values[
            static_cast<int64_t>(temporal_sources[local_chunk]) *
                kVectorsPerChunk +
            vector];
  }
}

enum class ValueSource : int {
  kCompatibility = 0,
  kCompactMiss = 1,
  kMappedHost = 2,
};

template <ValueSource Source>
__global__ void shadowkv_place_value_a100_kernel(
    const int8_t* __restrict__ component_kinds,
    const int32_t* __restrict__ source_slots,
    const int32_t* __restrict__ destination_slots,
    const int32_t* __restrict__ miss_ordinals,
    const int32_t* __restrict__ selected_chunk_ids,
    const int32_t* __restrict__ selected_lengths,
    const int32_t* __restrict__ plan_slots,
    int32_t* __restrict__ planner_error_codes,
    const uint4* __restrict__ temporal_key_values,
    const int32_t* __restrict__ value_miss_chunk_ids,
    const int32_t* __restrict__ value_miss_lengths,
    const int64_t* __restrict__ descriptor_generation,
    const uint8_t* __restrict__ descriptor_validity,
    int64_t expected_generation,
    int64_t temporal_chunks_per_component,
    int plan_capacity,
    const uint4* __restrict__ value_source,
    int prompt_chunk_capacity,
    int prompt_tokens,
    uint4* __restrict__ destination_key_values) {
  const int entry = blockIdx.x;
  const int selected = entry % kSelectedCapacity;
  const int head = entry / kSelectedCapacity;
  const int64_t plan_offset =
      static_cast<int64_t>(kKVHeads) * kSelectedCapacity + entry;
  const int8_t kind = component_kinds[plan_offset];
  const int source_slot = source_slots[plan_offset];
  const int destination_slot = destination_slots[plan_offset];
  const int miss_ordinal =
      miss_ordinals == nullptr ? -1 : miss_ordinals[plan_offset];
  const int selected_length = selected_lengths[head];
  const int plan_slot = plan_slots[head];
  const bool row_valid =
      planner_error_codes[head] == 0 && plan_slot >= 0 &&
      plan_slot < plan_capacity && selected_length >= 0 &&
      selected_length <= kSelectedCapacity;
  const bool active_ordinal = selected < selected_length;
  const int64_t expected_destination =
      ((static_cast<int64_t>(plan_capacity) + plan_slot) * kKVHeads + head) *
          kSelectedCapacity +
      selected;
  const bool hit_valid =
      active_ordinal && kind == kHit &&
      source_slot >= temporal_chunks_per_component &&
      source_slot < 2 * temporal_chunks_per_component &&
      (Source == ValueSource::kCompatibility || miss_ordinal == -1);
  bool miss_valid =
      active_ordinal && kind == kMiss && source_slot == -1;
  int source_chunk = selected;
  if constexpr (Source != ValueSource::kCompatibility) {
    const bool descriptor_ready =
        descriptor_validity[0] == 1 &&
        descriptor_generation[0] == expected_generation;
    const int miss_length = value_miss_lengths[head];
    miss_valid =
        miss_valid && descriptor_ready && miss_ordinal >= 0 &&
        miss_ordinal < miss_length && miss_length >= 0 &&
        miss_length <= kSelectedCapacity &&
        value_miss_chunk_ids[static_cast<int64_t>(head) * kSelectedCapacity +
                             miss_ordinal] ==
            selected_chunk_ids[static_cast<int64_t>(head) *
                                   kSelectedCapacity +
                               selected];
    source_chunk = miss_ordinal;
    if constexpr (Source == ValueSource::kMappedHost) {
      source_chunk = selected_chunk_ids[
          static_cast<int64_t>(head) * kSelectedCapacity + selected];
      miss_valid =
          miss_valid && source_chunk >= 0 &&
          source_chunk < prompt_chunk_capacity &&
          (static_cast<int64_t>(source_chunk) + 1) * kChunkSize <=
              prompt_tokens;
    }
  }
  const bool inactive_valid =
      kind == kInactive && source_slot == -1 && destination_slot == -1 &&
      (Source == ValueSource::kCompatibility || miss_ordinal == -1);
  const bool destination_valid = destination_slot == expected_destination;
  const bool active =
      row_valid && destination_valid && (hit_valid || miss_valid);
  const int64_t destination_vector =
      (static_cast<int64_t>(kKVHeads) * kSelectedCapacity + entry) *
          kVectorsPerChunk +
      threadIdx.x;
  if (!active) {
    destination_key_values[destination_vector] = uint4{0, 0, 0, 0};
    mark_invalid(
        planner_error_codes,
        head,
        row_valid && !inactive_valid);
    return;
  }
  if (hit_valid) {
    destination_key_values[destination_vector] =
        temporal_key_values[
            static_cast<int64_t>(source_slot) * kVectorsPerChunk +
            threadIdx.x];
    return;
  }
  int64_t source_vector = 0;
  if constexpr (Source == ValueSource::kCompatibility) {
    source_vector =
        (static_cast<int64_t>(kKVHeads) * kSelectedCapacity + entry) *
            kVectorsPerChunk +
        threadIdx.x;
  } else if constexpr (Source == ValueSource::kCompactMiss) {
    source_vector =
        (static_cast<int64_t>(head) * kSelectedCapacity + source_chunk) *
            kVectorsPerChunk +
        threadIdx.x;
  } else {
    source_vector =
        (static_cast<int64_t>(head) * prompt_chunk_capacity + source_chunk) *
            kVectorsPerChunk +
        threadIdx.x;
  }
  destination_key_values[destination_vector] = value_source[source_vector];
}

}  // namespace

void shadowkv_fused_key_a100(
    const at::Tensor& u,
    const at::Tensor& sv,
    const at::Tensor& cosine,
    const at::Tensor& sine,
    const at::Tensor& component_kinds,
    const at::Tensor& source_slots,
    const at::Tensor& destination_slots,
    const at::Tensor& miss_ordinals,
    const at::Tensor& selected_chunk_ids,
    const at::Tensor& selected_lengths,
    const at::Tensor& plan_slots,
    at::Tensor& planner_error_codes,
    const at::Tensor& temporal_key_values,
    int64_t plan_capacity,
    at::Tensor& destination_key_values) {
  check_tensor(u, "u", at::ScalarType::BFloat16, 2);
  check_tensor(sv, "sv", at::ScalarType::BFloat16, 3);
  check_tensor(cosine, "cosine", at::ScalarType::Float, 2);
  check_tensor(sine, "sine", at::ScalarType::Float, 2);
  check_tensor(miss_ordinals, "miss_ordinals", at::ScalarType::Int, 3);
  check_tensor(selected_chunk_ids, "selected_chunk_ids", at::ScalarType::Int, 2);
  const PlanGeometry geometry = check_plan_tensors(
      component_kinds,
      source_slots,
      destination_slots,
      selected_lengths,
      plan_slots,
      planner_error_codes,
      temporal_key_values,
      plan_capacity,
      destination_key_values);
  TORCH_CHECK(
      u.size(0) >= 1 && u.size(0) <= 8192 && u.size(1) == kRank,
      "u must have shape [1..8192, 160]");
  TORCH_CHECK(
      sv.sizes() == at::IntArrayRef({kKVHeads, kRank, kHeadDimension}),
      "sv must have shape [8, 160, 128]");
  TORCH_CHECK(
      cosine.size(0) >= u.size(0) && cosine.size(0) <= 8192 &&
          cosine.size(1) == kHalfHeadDimension && sine.sizes() == cosine.sizes(),
      "cosine and sine must share shape [u_tokens..8192, 64]");
  TORCH_CHECK(
      miss_ordinals.sizes() == component_kinds.sizes(),
      "miss_ordinals must have shape [2, 8, 256]");
  TORCH_CHECK(
      selected_chunk_ids.sizes() ==
          at::IntArrayRef({kKVHeads, kSelectedCapacity}),
      "selected_chunk_ids must have shape [8, 256]");
  const auto device = u.device();
  const at::Tensor* tensors[] = {
      &sv,
      &cosine,
      &sine,
      &component_kinds,
      &source_slots,
      &destination_slots,
      &miss_ordinals,
      &selected_chunk_ids,
      &selected_lengths,
      &plan_slots,
      &planner_error_codes,
      &temporal_key_values,
      &destination_key_values,
  };
  for (const at::Tensor* tensor : tensors) {
    TORCH_CHECK(
        tensor->device() == device,
        "all A100 fused-key tensors must share one CUDA device");
  }
  c10::cuda::CUDAGuard device_guard(device);
  check_sm80(u);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  shadowkv_fused_key_a100_kernel<<<kKVHeads * kKeyBlocksPerHead, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(u.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(sv.data_ptr<at::BFloat16>()),
      cosine.data_ptr<float>(),
      sine.data_ptr<float>(),
      component_kinds.data_ptr<int8_t>(),
      source_slots.data_ptr<int32_t>(),
      destination_slots.data_ptr<int32_t>(),
      miss_ordinals.data_ptr<int32_t>(),
      selected_chunk_ids.data_ptr<int32_t>(),
      selected_lengths.data_ptr<int32_t>(),
      plan_slots.data_ptr<int32_t>(),
      planner_error_codes.data_ptr<int32_t>(),
      reinterpret_cast<const uint4*>(
          temporal_key_values.data_ptr<at::BFloat16>()),
      static_cast<int>(u.size(0)),
      static_cast<int>(cosine.size(0)),
      geometry.temporal_chunks_per_component,
      static_cast<int>(plan_capacity),
      reinterpret_cast<uint4*>(
          destination_key_values.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void shadowkv_place_value_a100(
    const at::Tensor& component_kinds,
    const at::Tensor& source_slots,
    const at::Tensor& destination_slots,
    const at::Tensor& selected_lengths,
    const at::Tensor& plan_slots,
    at::Tensor& planner_error_codes,
    const at::Tensor& temporal_key_values,
    const at::Tensor& compatibility_key_values,
    int64_t plan_capacity,
    at::Tensor& destination_key_values) {
  check_tensor(
      compatibility_key_values,
      "compatibility_key_values",
      at::ScalarType::BFloat16,
      5);
  const PlanGeometry geometry = check_plan_tensors(
      component_kinds,
      source_slots,
      destination_slots,
      selected_lengths,
      plan_slots,
      planner_error_codes,
      temporal_key_values,
      plan_capacity,
      destination_key_values);
  TORCH_CHECK(
      compatibility_key_values.sizes() == destination_key_values.sizes(),
      "compatibility_key_values must have shape [2, 8, 256, 8, 128]");
  TORCH_CHECK(
      compatibility_key_values.device() == component_kinds.device(),
      "compatibility values must share the plan CUDA device");
  c10::cuda::CUDAGuard device_guard(component_kinds.device());
  check_sm80(component_kinds);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  shadowkv_place_value_a100_kernel<ValueSource::kCompatibility>
      <<<kKVHeads * kSelectedCapacity, kThreads, 0, stream>>>(
          component_kinds.data_ptr<int8_t>(),
          source_slots.data_ptr<int32_t>(),
          destination_slots.data_ptr<int32_t>(),
          nullptr,
          nullptr,
          selected_lengths.data_ptr<int32_t>(),
          plan_slots.data_ptr<int32_t>(),
          planner_error_codes.data_ptr<int32_t>(),
          reinterpret_cast<const uint4*>(
              temporal_key_values.data_ptr<at::BFloat16>()),
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          0,
          geometry.temporal_chunks_per_component,
          static_cast<int>(plan_capacity),
          reinterpret_cast<const uint4*>(
              compatibility_key_values.data_ptr<at::BFloat16>()),
          0,
          0,
          reinterpret_cast<uint4*>(
              destination_key_values.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void shadowkv_place_value_miss_only_a100(
    const at::Tensor& component_kinds,
    const at::Tensor& source_slots,
    const at::Tensor& destination_slots,
    const at::Tensor& miss_ordinals,
    const at::Tensor& selected_chunk_ids,
    const at::Tensor& selected_lengths,
    const at::Tensor& plan_slots,
    at::Tensor& planner_error_codes,
    const at::Tensor& temporal_key_values,
    const at::Tensor& value_miss_chunk_ids,
    const at::Tensor& value_miss_lengths,
    const at::Tensor& descriptor_generation,
    const at::Tensor& descriptor_validity,
    int64_t expected_generation,
    int64_t plan_capacity,
    const at::Tensor& value_miss_key_values,
    at::Tensor& destination_key_values) {
  check_tensor(miss_ordinals, "miss_ordinals", at::ScalarType::Int, 3);
  check_tensor(selected_chunk_ids, "selected_chunk_ids", at::ScalarType::Int, 2);
  check_tensor(value_miss_chunk_ids, "value_miss_chunk_ids", at::ScalarType::Int, 2);
  check_tensor(value_miss_lengths, "value_miss_lengths", at::ScalarType::Int, 1);
  check_tensor(descriptor_generation, "descriptor_generation", at::ScalarType::Long, 1);
  check_tensor(descriptor_validity, "descriptor_validity", at::ScalarType::Byte, 1);
  check_tensor(value_miss_key_values, "value_miss_key_values", at::ScalarType::BFloat16, 4);
  const PlanGeometry geometry = check_plan_tensors(
      component_kinds,
      source_slots,
      destination_slots,
      selected_lengths,
      plan_slots,
      planner_error_codes,
      temporal_key_values,
      plan_capacity,
      destination_key_values);
  TORCH_CHECK(
      miss_ordinals.sizes() == component_kinds.sizes(),
      "miss_ordinals must have shape [2, 8, 256]");
  const auto compact_shape =
      at::IntArrayRef({kKVHeads, kSelectedCapacity});
  TORCH_CHECK(
      selected_chunk_ids.sizes() == compact_shape &&
          value_miss_chunk_ids.sizes() == compact_shape,
      "selected and value-miss chunk ids must have shape [8, 256]");
  TORCH_CHECK(
      value_miss_lengths.sizes() == at::IntArrayRef({kKVHeads}) &&
          descriptor_generation.sizes() == at::IntArrayRef({1}) &&
          descriptor_validity.sizes() == at::IntArrayRef({1}),
      "value descriptor rows or generation have invalid shapes");
  TORCH_CHECK(
      value_miss_key_values.sizes() ==
          at::IntArrayRef({kKVHeads, kSelectedCapacity, kChunkSize, kHeadDimension}),
      "value_miss_key_values must have shape [8, 256, 8, 128]");
  TORCH_CHECK(expected_generation >= 0, "expected_generation must be nonnegative");
  const auto device = component_kinds.device();
  const at::Tensor* tensors[] = {
      &miss_ordinals,
      &selected_chunk_ids,
      &value_miss_chunk_ids,
      &value_miss_lengths,
      &descriptor_generation,
      &descriptor_validity,
      &value_miss_key_values,
  };
  for (const at::Tensor* tensor : tensors) {
    TORCH_CHECK(
        tensor->device() == device,
        "all A100 value-miss tensors must share one CUDA device");
  }
  c10::cuda::CUDAGuard device_guard(device);
  check_sm80(component_kinds);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  shadowkv_place_value_a100_kernel<ValueSource::kCompactMiss>
      <<<kKVHeads * kSelectedCapacity, kThreads, 0, stream>>>(
          component_kinds.data_ptr<int8_t>(),
          source_slots.data_ptr<int32_t>(),
          destination_slots.data_ptr<int32_t>(),
          miss_ordinals.data_ptr<int32_t>(),
          selected_chunk_ids.data_ptr<int32_t>(),
          selected_lengths.data_ptr<int32_t>(),
          plan_slots.data_ptr<int32_t>(),
          planner_error_codes.data_ptr<int32_t>(),
          reinterpret_cast<const uint4*>(
              temporal_key_values.data_ptr<at::BFloat16>()),
          value_miss_chunk_ids.data_ptr<int32_t>(),
          value_miss_lengths.data_ptr<int32_t>(),
          descriptor_generation.data_ptr<int64_t>(),
          descriptor_validity.data_ptr<uint8_t>(),
          expected_generation,
          geometry.temporal_chunks_per_component,
          static_cast<int>(plan_capacity),
          reinterpret_cast<const uint4*>(
              value_miss_key_values.data_ptr<at::BFloat16>()),
          0,
          0,
          reinterpret_cast<uint4*>(
              destination_key_values.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void shadowkv_place_value_mapped_host_a100(
    const at::Tensor& component_kinds,
    const at::Tensor& source_slots,
    const at::Tensor& destination_slots,
    const at::Tensor& miss_ordinals,
    const at::Tensor& selected_chunk_ids,
    const at::Tensor& selected_lengths,
    const at::Tensor& plan_slots,
    at::Tensor& planner_error_codes,
    const at::Tensor& temporal_key_values,
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
  check_tensor(miss_ordinals, "miss_ordinals", at::ScalarType::Int, 3);
  check_tensor(selected_chunk_ids, "selected_chunk_ids", at::ScalarType::Int, 2);
  check_tensor(value_miss_chunk_ids, "value_miss_chunk_ids", at::ScalarType::Int, 2);
  check_tensor(value_miss_lengths, "value_miss_lengths", at::ScalarType::Int, 1);
  check_tensor(descriptor_generation, "descriptor_generation", at::ScalarType::Long, 1);
  check_tensor(descriptor_validity, "descriptor_validity", at::ScalarType::Byte, 1);
  const PlanGeometry geometry = check_plan_tensors(
      component_kinds,
      source_slots,
      destination_slots,
      selected_lengths,
      plan_slots,
      planner_error_codes,
      temporal_key_values,
      plan_capacity,
      destination_key_values);
  TORCH_CHECK(
      miss_ordinals.sizes() == component_kinds.sizes(),
      "miss_ordinals must have shape [2, 8, 256]");
  const auto compact_shape =
      at::IntArrayRef({kKVHeads, kSelectedCapacity});
  TORCH_CHECK(
      selected_chunk_ids.sizes() == compact_shape &&
          value_miss_chunk_ids.sizes() == compact_shape,
      "selected and value-miss chunk ids must have shape [8, 256]");
  TORCH_CHECK(
      value_miss_lengths.sizes() == at::IntArrayRef({kKVHeads}) &&
          descriptor_generation.sizes() == at::IntArrayRef({1}) &&
          descriptor_validity.sizes() == at::IntArrayRef({1}),
      "value descriptor rows or generation have invalid shapes");
  TORCH_CHECK(
      mapped_host_pointer > 0 && mapped_host_pointer % kVectorBytes == 0,
      "mapped host pointer is invalid");
  TORCH_CHECK(
      prompt_chunk_capacity >= 1 && prompt_tokens >= 1 &&
          prompt_tokens <= prompt_chunk_capacity * kChunkSize,
      "mapped host prompt bounds are invalid");
  TORCH_CHECK(
      prompt_chunk_capacity <=
          std::numeric_limits<int64_t>::max() / kKVHeads / kChunkBytes,
      "mapped host byte range overflows int64");
  TORCH_CHECK(
      mapped_host_bytes >= kKVHeads * prompt_chunk_capacity * kChunkBytes,
      "mapped host byte range is smaller than its declared shape");
  TORCH_CHECK(expected_generation >= 0, "expected_generation must be nonnegative");
  const auto device = component_kinds.device();
  const at::Tensor* tensors[] = {
      &miss_ordinals,
      &selected_chunk_ids,
      &value_miss_chunk_ids,
      &value_miss_lengths,
      &descriptor_generation,
      &descriptor_validity,
  };
  for (const at::Tensor* tensor : tensors) {
    TORCH_CHECK(
        tensor->device() == device,
        "all A100 mapped-value tensors must share one CUDA device");
  }
  c10::cuda::CUDAGuard device_guard(device);
  check_sm80(component_kinds);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  shadowkv_place_value_a100_kernel<ValueSource::kMappedHost>
      <<<kKVHeads * kSelectedCapacity, kThreads, 0, stream>>>(
          component_kinds.data_ptr<int8_t>(),
          source_slots.data_ptr<int32_t>(),
          destination_slots.data_ptr<int32_t>(),
          miss_ordinals.data_ptr<int32_t>(),
          selected_chunk_ids.data_ptr<int32_t>(),
          selected_lengths.data_ptr<int32_t>(),
          plan_slots.data_ptr<int32_t>(),
          planner_error_codes.data_ptr<int32_t>(),
          reinterpret_cast<const uint4*>(
              temporal_key_values.data_ptr<at::BFloat16>()),
          value_miss_chunk_ids.data_ptr<int32_t>(),
          value_miss_lengths.data_ptr<int32_t>(),
          descriptor_generation.data_ptr<int64_t>(),
          descriptor_validity.data_ptr<uint8_t>(),
          expected_generation,
          geometry.temporal_chunks_per_component,
          static_cast<int>(plan_capacity),
          reinterpret_cast<const uint4*>(
              static_cast<uintptr_t>(mapped_host_pointer)),
          static_cast<int>(prompt_chunk_capacity),
          static_cast<int>(prompt_tokens),
          reinterpret_cast<uint4*>(
              destination_key_values.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
