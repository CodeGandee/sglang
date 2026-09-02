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
#include <c10/cuda/CUDAStream.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/all.h>

#include "cutlass/bfloat16.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <mutex>

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
constexpr int kValueEntries = kKVHeads * kSelectedCapacity;
constexpr int kThrottledMappedValueBlocks = 512;
constexpr int kExactGemmAlgorithm = 6;
constexpr uint32_t kExactGemmTile = CUBLASLT_MATMUL_TILE_128x64;
constexpr uint32_t kExactGemmStages = CUBLASLT_MATMUL_STAGES_64x3;

// This fixed SM80 shape is part of the Llama 3.1 8B specialization contract.
// It uses the CUTLASS headers already fetched by the AOT build and does not
// depend on the external ShadowKV reference checkout at build or run time.
using ShadowKVCutlassElement = cutlass::bfloat16_t;
using ShadowKVCutlassAccumulator = float;
using ShadowKVCutlassOutputOp =
    cutlass::epilogue::thread::LinearCombination<
        ShadowKVCutlassElement,
        128 / cutlass::sizeof_bits<ShadowKVCutlassElement>::value,
        ShadowKVCutlassAccumulator,
        ShadowKVCutlassAccumulator>;
using ShadowKVMissGemmDefault = cutlass::gemm::kernel::DefaultGemm<
    ShadowKVCutlassElement,
    cutlass::layout::RowMajor,
    8,
    ShadowKVCutlassElement,
    cutlass::layout::RowMajor,
    8,
    ShadowKVCutlassElement,
    cutlass::layout::RowMajor,
    ShadowKVCutlassAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 64>,
    cutlass::gemm::GemmShape<64, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    ShadowKVCutlassOutputOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    false,
    cutlass::arch::OpMultiplyAdd,
    cutlass::gemm::SharedMemoryClearOption::kNone,
    false,
    false,
    true>;
using ShadowKVMissGemmKernel = typename ShadowKVMissGemmDefault::GemmKernel;
constexpr int kMissGemmThreads = ShadowKVMissGemmKernel::kThreadCount;
constexpr size_t kMissGemmScatterOffset =
    (sizeof(ShadowKVMissGemmKernel::SharedStorage) + alignof(int32_t) - 1) &
    ~(alignof(int32_t) - 1);
constexpr size_t kMissGemmSharedBytes =
    kMissGemmScatterOffset + (kRowsPerKeyBlock + 1) * sizeof(int32_t);

static_assert(kVectorsPerChunk <= kThreads);
static_assert(kThreads <= 1024 && kThreads % 32 == 0);
static_assert(kSelectedCapacity % kChunksPerKeyBlock == 0);
static_assert(kRowsPerKeyBlock == 128 && kThreads == 256);
static_assert(kValueEntries % kThrottledMappedValueBlocks == 0);
static_assert(kMissGemmThreads == 128);

struct MissCounts {
  int32_t values[kKVHeads];
};

struct ExactMissGemmRuntime {
  std::mutex mutex;
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDescOpaque_t operation{};
  cublasLtMatmulAlgo_t algorithm{};
  int device_index = -1;
  bool prepared = false;
};

ExactMissGemmRuntime& exact_miss_gemm_runtime() {
  static ExactMissGemmRuntime runtime;
  return runtime;
}

void check_cublas(cublasStatus_t status, const char* operation) {
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      operation,
      " failed with cuBLAS status ",
      static_cast<int>(status));
}

void configure_exact_matrix_layout(
    cublasLtMatrixLayout_t layout,
    int32_t batch_count,
    int64_t batch_stride) {
  const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  check_cublas(
      cublasLtMatrixLayoutSetAttribute(
          layout,
          CUBLASLT_MATRIX_LAYOUT_ORDER,
          &order,
          sizeof(order)),
      "set exact GEMM row-major layout");
  check_cublas(
      cublasLtMatrixLayoutSetAttribute(
          layout,
          CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
          &batch_count,
          sizeof(batch_count)),
      "set exact GEMM batch count");
  check_cublas(
      cublasLtMatrixLayoutSetAttribute(
          layout,
          CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
          &batch_stride,
          sizeof(batch_stride)),
      "set exact GEMM batch stride");
}

struct ExactMatrixLayouts {
  cublasLtMatrixLayoutOpaque_t a{};
  cublasLtMatrixLayoutOpaque_t b{};
  cublasLtMatrixLayoutOpaque_t c{};
  cublasLtMatrixLayoutOpaque_t d{};

  explicit ExactMatrixLayouts(int rows) {
    check_cublas(
        cublasLtMatrixLayoutInit(&a, CUDA_R_16BF, rows, kRank, kRank),
        "initialize exact GEMM A layout");
    check_cublas(
        cublasLtMatrixLayoutInit(
            &b, CUDA_R_16BF, kRank, kHeadDimension, kHeadDimension),
        "initialize exact GEMM B layout");
    check_cublas(
        cublasLtMatrixLayoutInit(
            &c, CUDA_R_16BF, rows, kHeadDimension, kHeadDimension),
        "initialize exact GEMM C layout");
    check_cublas(
        cublasLtMatrixLayoutInit(
            &d, CUDA_R_16BF, rows, kHeadDimension, kHeadDimension),
        "initialize exact GEMM D layout");
    configure_exact_matrix_layout(
        &a,
        kKVHeads,
        static_cast<int64_t>(kSelectedCapacity) * kChunkSize * kRank);
    configure_exact_matrix_layout(
        &b,
        kKVHeads,
        static_cast<int64_t>(kRank) * kHeadDimension);
    configure_exact_matrix_layout(
        &c,
        kKVHeads,
        static_cast<int64_t>(kSelectedCapacity) * kChunkSize *
            kHeadDimension);
    configure_exact_matrix_layout(
        &d,
        kKVHeads,
        static_cast<int64_t>(kSelectedCapacity) * kChunkSize *
            kHeadDimension);
  }
};

void prepare_exact_miss_gemm(int device_index) {
  ExactMissGemmRuntime& runtime = exact_miss_gemm_runtime();
  std::lock_guard<std::mutex> guard(runtime.mutex);
  if (runtime.prepared) {
    TORCH_CHECK(
        runtime.device_index == device_index,
        "exact A100 miss GEMM was prepared on a different CUDA device");
    return;
  }
  check_cublas(cublasLtCreate(&runtime.handle), "create exact GEMM handle");
  runtime.device_index = device_index;
  check_cublas(
      cublasLtMatmulDescInit(
          &runtime.operation, CUBLAS_COMPUTE_32F, CUDA_R_32F),
      "initialize exact GEMM operation");
  check_cublas(
      cublasLtMatmulAlgoInit(
          runtime.handle,
          CUBLAS_COMPUTE_32F,
          CUDA_R_32F,
          CUDA_R_16BF,
          CUDA_R_16BF,
          CUDA_R_16BF,
          CUDA_R_16BF,
          kExactGemmAlgorithm,
          &runtime.algorithm),
      "initialize exact GEMM algorithm");
  const int32_t split_k = 1;
  const uint32_t reduction_scheme = CUBLASLT_REDUCTION_SCHEME_NONE;
  const uint32_t swizzle = 0;
  const uint32_t custom_option = 0;
  check_cublas(
      cublasLtMatmulAlgoConfigSetAttribute(
          &runtime.algorithm,
          CUBLASLT_ALGO_CONFIG_TILE_ID,
          &kExactGemmTile,
          sizeof(kExactGemmTile)),
      "set exact GEMM tile");
  check_cublas(
      cublasLtMatmulAlgoConfigSetAttribute(
          &runtime.algorithm,
          CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
          &split_k,
          sizeof(split_k)),
      "set exact GEMM split-K");
  check_cublas(
      cublasLtMatmulAlgoConfigSetAttribute(
          &runtime.algorithm,
          CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
          &reduction_scheme,
          sizeof(reduction_scheme)),
      "set exact GEMM reduction");
  check_cublas(
      cublasLtMatmulAlgoConfigSetAttribute(
          &runtime.algorithm,
          CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
          &swizzle,
          sizeof(swizzle)),
      "set exact GEMM swizzle");
  check_cublas(
      cublasLtMatmulAlgoConfigSetAttribute(
          &runtime.algorithm,
          CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
          &custom_option,
          sizeof(custom_option)),
      "set exact GEMM custom option");
  check_cublas(
      cublasLtMatmulAlgoConfigSetAttribute(
          &runtime.algorithm,
          CUBLASLT_ALGO_CONFIG_STAGES_ID,
          &kExactGemmStages,
          sizeof(kExactGemmStages)),
      "set exact GEMM stages");
  for (int miss_chunks = 1; miss_chunks <= kSelectedCapacity; ++miss_chunks) {
    ExactMatrixLayouts layouts(miss_chunks * kChunkSize);
    cublasLtMatmulHeuristicResult_t checked{};
    check_cublas(
        cublasLtMatmulAlgoCheck(
            runtime.handle,
            &runtime.operation,
            &layouts.a,
            &layouts.b,
            &layouts.c,
            &layouts.d,
            &runtime.algorithm,
            &checked),
        "validate exact GEMM miss bucket");
    TORCH_CHECK(
        checked.workspaceSize == 0,
        "exact A100 miss GEMM unexpectedly requires workspace");
  }
  runtime.prepared = true;
}

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

void check_alignment(
    const at::Tensor& tensor,
    const char* name,
    uintptr_t alignment) {
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignment == 0,
      name,
      " must be ",
      alignment,
      "-byte aligned");
}

void check_sm80_device(int device_index) {
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device_index));
  TORCH_CHECK(
      properties.major == 8 && properties.minor == 0,
      "A100 fused ShadowKV kernels require compute capability 8.0; found ",
      properties.major,
      ".",
      properties.minor);
}

void check_sm80(const at::Tensor& tensor) {
  check_sm80_device(tensor.get_device());
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
  check_alignment(temporal_key_values, "temporal_key_values", kVectorBytes);
  check_alignment(destination_key_values, "destination_key_values", kVectorBytes);

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

__global__ void shadowkv_prepare_key_bmm_a100_kernel(
    const __nv_bfloat16* __restrict__ u,
    const int8_t* __restrict__ component_kinds,
    const int32_t* __restrict__ source_slots,
    const int32_t* __restrict__ destination_slots,
    const int32_t* __restrict__ miss_ordinals,
    const int32_t* __restrict__ selected_chunk_ids,
    const int32_t* __restrict__ selected_lengths,
    const int32_t* __restrict__ plan_slots,
    int32_t* __restrict__ planner_error_codes,
    int u_tokens,
    int rope_rows,
    int64_t temporal_chunks_per_component,
    int plan_capacity,
    __nv_bfloat16* __restrict__ gathered_u) {
  const int entry = blockIdx.x;
  const int selected = entry % kSelectedCapacity;
  const int head = entry / kSelectedCapacity;
  const int64_t plan_offset =
      static_cast<int64_t>(head) * kSelectedCapacity + selected;
  const int8_t kind = component_kinds[plan_offset];
  const int source_slot = source_slots[plan_offset];
  const int destination_slot = destination_slots[plan_offset];
  const int miss_ordinal = miss_ordinals[plan_offset];
  const int selected_chunk = selected_chunk_ids[plan_offset];
  const int selected_length = selected_lengths[head];
  const int plan_slot = plan_slots[head];
  const bool plan_row_valid =
      planner_error_codes[head] == 0 && plan_slot >= 0 &&
      plan_slot < plan_capacity && selected_length >= 0 &&
      selected_length <= kSelectedCapacity;
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
      (static_cast<int64_t>(selected_chunk) + 1) * kChunkSize <= u_tokens &&
      (static_cast<int64_t>(selected_chunk) + 1) * kChunkSize <= rope_rows;
  const bool inactive_valid =
      !active_ordinal && kind == kInactive && source_slot == -1 &&
      destination_slot == -1 && miss_ordinal == -1;
  const bool destination_valid =
      active_ordinal && destination_slot == expected_destination;
  const bool entry_valid =
      plan_row_valid &&
      (inactive_valid || (destination_valid && (hit_valid || miss_valid)));
  if (threadIdx.x == 0 && plan_row_valid && !entry_valid) {
    atomicCAS(
        planner_error_codes + head,
        0,
        kInvalidPlacementDescriptor);
  }

  const int64_t destination_row =
      (static_cast<int64_t>(head) * kSelectedCapacity + selected) *
      kChunkSize;
  for (int index = threadIdx.x;
       index < kChunkSize * kRank;
       index += blockDim.x) {
    const int token = index / kRank;
    const int rank = index - token * kRank;
    gathered_u[(destination_row + token) * kRank + rank] =
        entry_valid && miss_valid
        ? u[(static_cast<int64_t>(selected_chunk) * kChunkSize + token) *
                kRank +
            rank]
        : __float2bfloat16_rn(0.0f);
  }
}

__global__ void shadowkv_validate_key_miss_plan_a100_kernel(
    const int8_t* __restrict__ component_kinds,
    const int32_t* __restrict__ source_slots,
    const int32_t* __restrict__ destination_slots,
    const int32_t* __restrict__ miss_ordinals,
    const int32_t* __restrict__ selected_chunk_ids,
    const int32_t* __restrict__ selected_lengths,
    const int32_t* __restrict__ plan_slots,
    int32_t* __restrict__ planner_error_codes,
    int u_tokens,
    int rope_rows,
    int64_t temporal_chunks_per_component,
    int plan_capacity,
    int32_t* __restrict__ mapped_miss_counts) {
  const int head = blockIdx.x;
  const int selected = threadIdx.x;
  __shared__ int32_t seen_ordinals[kSelectedCapacity];
  __shared__ int32_t hit_count;
  __shared__ int32_t miss_count;

  seen_ordinals[selected] = 0;
  if (selected == 0) {
    hit_count = 0;
    miss_count = 0;
  }
  __syncthreads();

  const int64_t plan_offset =
      static_cast<int64_t>(head) * kSelectedCapacity + selected;
  const int8_t kind = component_kinds[plan_offset];
  const int source_slot = source_slots[plan_offset];
  const int destination_slot = destination_slots[plan_offset];
  const int miss_ordinal = miss_ordinals[plan_offset];
  const int selected_chunk = selected_chunk_ids[plan_offset];
  const int selected_length = selected_lengths[head];
  const int plan_slot = plan_slots[head];
  const bool plan_row_valid =
      planner_error_codes[head] == 0 && plan_slot >= 0 &&
      plan_slot < plan_capacity && selected_length >= 0 &&
      selected_length <= kSelectedCapacity;
  if (selected == 0 && planner_error_codes[head] == 0 && !plan_row_valid) {
    atomicCAS(
        planner_error_codes + head,
        0,
        kInvalidPlacementDescriptor);
  }
  const bool active_ordinal = selected < selected_length;
  const int64_t expected_destination =
      (static_cast<int64_t>(plan_slot) * kKVHeads + head) *
          kSelectedCapacity +
      selected;
  const bool inactive_valid =
      kind == kInactive && source_slot == -1 && destination_slot == -1 &&
      miss_ordinal == -1;
  const bool hit_valid =
      active_ordinal && kind == kHit && source_slot >= 0 &&
      source_slot < temporal_chunks_per_component && miss_ordinal == -1 &&
      destination_slot == expected_destination;
  const bool miss_valid =
      active_ordinal && kind == kMiss && source_slot == -1 &&
      miss_ordinal >= 0 && miss_ordinal < kSelectedCapacity &&
      selected_chunk >= 0 &&
      (static_cast<int64_t>(selected_chunk) + 1) * kChunkSize <= u_tokens &&
      (static_cast<int64_t>(selected_chunk) + 1) * kChunkSize <= rope_rows &&
      destination_slot == expected_destination;
  const bool entry_valid =
      plan_row_valid && (inactive_valid || hit_valid || miss_valid) &&
      (active_ordinal || inactive_valid);

  if (!entry_valid) {
    if (plan_row_valid) {
      atomicCAS(
          planner_error_codes + head,
          0,
          kInvalidPlacementDescriptor);
    }
  } else if (hit_valid) {
    atomicAdd(&hit_count, 1);
  } else if (miss_valid) {
    atomicAdd(&miss_count, 1);
    if (atomicCAS(seen_ordinals + miss_ordinal, 0, 1) != 0) {
      atomicCAS(
          planner_error_codes + head,
          0,
          kInvalidPlacementDescriptor);
    }
  }
  __syncthreads();

  if (plan_row_valid) {
    const bool ordinal_present = seen_ordinals[selected] != 0;
    const bool ordinal_expected = selected < miss_count;
    if (ordinal_present != ordinal_expected ||
        hit_count + miss_count > selected_length) {
      atomicCAS(
          planner_error_codes + head,
          0,
          kInvalidPlacementDescriptor);
    }
  }
  __syncthreads();
  if (planner_error_codes[head] != 0) {
    return;
  }
  if (selected == 0 && mapped_miss_counts != nullptr) {
    mapped_miss_counts[head] = miss_count;
    __threadfence_system();
  }
}

__global__ void shadowkv_materialize_key_misses_a100_kernel(
    const __nv_bfloat16* __restrict__ u,
    const int8_t* __restrict__ component_kinds,
    const int32_t* __restrict__ source_slots,
    const int32_t* __restrict__ miss_ordinals,
    const int32_t* __restrict__ selected_chunk_ids,
    const int32_t* __restrict__ planner_error_codes,
    const uint4* __restrict__ temporal_key_values,
    __nv_bfloat16* __restrict__ gathered_u,
    uint4* __restrict__ destination_key_values) {
  const int head = blockIdx.x / kSelectedCapacity;
  const int selected = blockIdx.x - head * kSelectedCapacity;
  if (planner_error_codes[head] != 0) {
    return;
  }
  const int64_t plan_offset =
      static_cast<int64_t>(head) * kSelectedCapacity + selected;
  const int8_t kind = component_kinds[plan_offset];

  for (int vector = threadIdx.x;
       vector < kVectorsPerChunk;
       vector += blockDim.x) {
    const int64_t destination_vector = plan_offset * kVectorsPerChunk + vector;
    if (kind == kHit) {
      destination_key_values[destination_vector] =
          temporal_key_values[
              static_cast<int64_t>(source_slots[plan_offset]) *
                  kVectorsPerChunk +
              vector];
    } else if (kind == kInactive) {
      destination_key_values[destination_vector] = uint4{0, 0, 0, 0};
    }
  }

  if (kind != kMiss) {
    return;
  }
  constexpr int kGatherElementsPerSelected = kChunkSize * kRank;
  for (int local = threadIdx.x;
       local < kGatherElementsPerSelected;
       local += blockDim.x) {
    const int token = local / kRank;
    const int rank = local - token * kRank;
    const int ordinal = miss_ordinals[plan_offset];
    const int chunk = selected_chunk_ids[plan_offset];
    gathered_u[
        ((static_cast<int64_t>(head) * kSelectedCapacity + ordinal) *
             kChunkSize +
         token) *
            kRank +
        rank] =
        u[(static_cast<int64_t>(chunk) * kChunkSize + token) * kRank + rank];
  }
}

__global__ void shadowkv_reconstruct_key_misses_a100_kernel(
    const ShadowKVCutlassElement* __restrict__ gathered_u,
    const ShadowKVCutlassElement* __restrict__ sv,
    const float* __restrict__ cosine,
    const float* __restrict__ sine,
    const int8_t* __restrict__ component_kinds,
    const int32_t* __restrict__ miss_ordinals,
    const int32_t* __restrict__ selected_chunk_ids,
    int32_t* __restrict__ planner_error_codes,
    ShadowKVCutlassElement* __restrict__ destination_key_values) {
  extern __shared__ __align__(16) unsigned char shared_bytes[];
  auto* shared_storage =
      reinterpret_cast<ShadowKVMissGemmKernel::SharedStorage*>(shared_bytes);
  int32_t* scatter_rows =
      reinterpret_cast<int32_t*>(shared_bytes + kMissGemmScatterOffset);
  int32_t* shared_miss_count = scatter_rows + kRowsPerKeyBlock;

  const int head = blockIdx.x;
  const int key_block = blockIdx.y;
  const int miss_base = key_block * kChunksPerKeyBlock;
  if (threadIdx.x == 0) {
    *shared_miss_count = 0;
  }
  if (threadIdx.x < kRowsPerKeyBlock) {
    scatter_rows[threadIdx.x] = -1;
  }
  __syncthreads();
  if (planner_error_codes[head] != 0) {
    return;
  }

  for (int selected = threadIdx.x;
       selected < kSelectedCapacity;
       selected += blockDim.x) {
    const int64_t plan_offset =
        static_cast<int64_t>(head) * kSelectedCapacity + selected;
    if (component_kinds[plan_offset] == kMiss) {
      atomicAdd(shared_miss_count, 1);
    }
  }
  __syncthreads();
  const int miss_count = *shared_miss_count;
  if (miss_base >= miss_count) {
    return;
  }
  const int active_chunks = min(kChunksPerKeyBlock, miss_count - miss_base);
  const int active_rows = active_chunks * kChunkSize;

  for (int selected = threadIdx.x;
       selected < kSelectedCapacity;
       selected += blockDim.x) {
    const int64_t plan_offset =
        static_cast<int64_t>(head) * kSelectedCapacity + selected;
    if (component_kinds[plan_offset] == kMiss) {
      const int ordinal = miss_ordinals[plan_offset];
      if (ordinal < miss_base || ordinal >= miss_base + active_chunks) {
        continue;
      }
      const int local_chunk = ordinal - miss_base;
#pragma unroll
      for (int token = 0; token < kChunkSize; ++token) {
        scatter_rows[local_chunk * kChunkSize + token] =
            selected * kChunkSize + token;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x < active_rows && scatter_rows[threadIdx.x] < 0) {
    atomicCAS(
        planner_error_codes + head,
        0,
        kInvalidPlacementDescriptor);
  }
  __syncthreads();
  if (planner_error_codes[head] != 0) {
    return;
  }

  using Mma = typename ShadowKVMissGemmKernel::Mma;
  using Epilogue = typename ShadowKVMissGemmKernel::Epilogue;
  const int thread = threadIdx.x;
  const int warp = cutlass::canonical_warp_idx_sync();
  const int lane = thread & 31;
  const cutlass::MatrixCoord problem_a(active_rows, kRank);
  const cutlass::MatrixCoord origin(0, 0);
  ShadowKVCutlassElement* head_destination =
      destination_key_values +
      static_cast<int64_t>(head) * kSelectedCapacity * kChunkSize *
          kHeadDimension;
  for (int column_base = 0;
       column_base < kHeadDimension;
       column_base += Mma::Shape::kN) {
    const cutlass::MatrixCoord problem_b(kRank, Mma::Shape::kN);
    const cutlass::MatrixCoord problem_output(active_rows, Mma::Shape::kN);
    typename Mma::IteratorA::Params params_a{
        cutlass::layout::RowMajor(kRank)};
    typename Mma::IteratorA iterator_a(
        params_a,
        const_cast<ShadowKVCutlassElement*>(
            gathered_u +
            (static_cast<int64_t>(head) * kSelectedCapacity + miss_base) *
                kChunkSize * kRank),
        problem_a,
        thread,
        origin);
    typename Mma::IteratorB::Params params_b{
        cutlass::layout::RowMajor(kHeadDimension)};
    typename Mma::IteratorB iterator_b(
        params_b,
        const_cast<ShadowKVCutlassElement*>(
            sv + static_cast<int64_t>(head) * kRank * kHeadDimension +
            column_base),
        problem_b,
        thread,
        origin);
    Mma mma(shared_storage->main_loop, thread, warp, lane);
    typename Mma::FragmentC accumulators;
    accumulators.clear();
    mma(
        (kRank + Mma::Shape::kK - 1) / Mma::Shape::kK,
        accumulators,
        iterator_a,
        iterator_b,
        accumulators);

    ShadowKVCutlassOutputOp output_op(
        typename ShadowKVCutlassOutputOp::Params(1.0f, 0.0f));
    typename Epilogue::OutputTileIterator::Params output_params{
        cutlass::layout::RowMajor(kHeadDimension)};
    typename Epilogue::OutputTileIterator destination_iterator(
        output_params,
        head_destination + column_base,
        problem_output,
        thread,
        origin,
        scatter_rows);
    typename Epilogue::OutputTileIterator source_iterator(
        output_params,
        head_destination + column_base,
        problem_output,
        thread,
        origin,
        scatter_rows);
    Epilogue epilogue(shared_storage->epilogue, thread, warp, lane);
    epilogue(output_op, destination_iterator, accumulators, source_iterator);
    __syncthreads();
  }

  __nv_bfloat16* destination =
      reinterpret_cast<__nv_bfloat16*>(destination_key_values);
  for (int pair = threadIdx.x;
       pair < active_rows * kHalfHeadDimension;
       pair += blockDim.x) {
    const int local_row = pair / kHalfHeadDimension;
    const int frequency = pair - local_row * kHalfHeadDimension;
    const int destination_row = scatter_rows[local_row];
    const int selected_slot = destination_row / kChunkSize;
    const int token = destination_row - selected_slot * kChunkSize;
    const int selected_chunk = selected_chunk_ids[
        static_cast<int64_t>(head) * kSelectedCapacity + selected_slot];
    const int64_t position =
        static_cast<int64_t>(selected_chunk) * kChunkSize + token;
    const int64_t destination_base =
        (static_cast<int64_t>(head) * kSelectedCapacity * kChunkSize +
         destination_row) *
        kHeadDimension;
    const float first =
        __bfloat162float(destination[destination_base + frequency]);
    const float second = __bfloat162float(
        destination[destination_base + kHalfHeadDimension + frequency]);
    const int64_t frequency_offset =
        position * kHalfHeadDimension + frequency;
    const float cosine_value = cosine[frequency_offset];
    const float sine_value = sine[frequency_offset];
    destination[destination_base + frequency] = __float2bfloat16_rn(
        __fadd_rn(
            __fmul_rn(first, cosine_value),
            __fmul_rn(-second, sine_value)));
    destination[destination_base + kHalfHeadDimension + frequency] =
        __float2bfloat16_rn(
            __fadd_rn(
                __fmul_rn(second, cosine_value),
                __fmul_rn(first, sine_value)));
  }
}

__global__ void shadowkv_zero_compact_key_padding_a100_kernel(
    __nv_bfloat16* __restrict__ gathered_u,
    MissCounts miss_counts,
    int maximum_miss_chunks) {
  const int head = blockIdx.x;
  const int first_element = miss_counts.values[head] * kChunkSize * kRank;
  const int end_element = maximum_miss_chunks * kChunkSize * kRank;
  for (int element = first_element + blockIdx.y * blockDim.x + threadIdx.x;
       element < end_element;
       element += gridDim.y * blockDim.x) {
    gathered_u[
        static_cast<int64_t>(head) * kSelectedCapacity * kChunkSize * kRank +
        element] = __float2bfloat16_rn(0.0f);
  }
}

__global__ void shadowkv_finalize_compact_key_misses_a100_kernel(
    const __nv_bfloat16* __restrict__ reconstructed_misses,
    const float* __restrict__ cosine,
    const float* __restrict__ sine,
    const int8_t* __restrict__ component_kinds,
    const int32_t* __restrict__ miss_ordinals,
    const int32_t* __restrict__ selected_chunk_ids,
    __nv_bfloat16* __restrict__ destination_key_values) {
  constexpr int kPairsPerSelected = kChunkSize * kHalfHeadDimension;
  constexpr int kPairsPerBlock = kChunksPerKeyBlock * kPairsPerSelected;
  const int head = blockIdx.x;
  const int selected_base = blockIdx.y * kChunksPerKeyBlock;
  for (int pair = threadIdx.x; pair < kPairsPerBlock; pair += blockDim.x) {
    const int local_selected = pair / kPairsPerSelected;
    const int local_pair = pair - local_selected * kPairsPerSelected;
    const int token = local_pair / kHalfHeadDimension;
    const int frequency = local_pair - token * kHalfHeadDimension;
    const int selected = selected_base + local_selected;
    const int64_t plan_offset =
        static_cast<int64_t>(head) * kSelectedCapacity + selected;
    if (component_kinds[plan_offset] != kMiss) {
      continue;
    }
    const int ordinal = miss_ordinals[plan_offset];
    const int64_t source_base =
        (static_cast<int64_t>(head) * kSelectedCapacity * kChunkSize +
         ordinal * kChunkSize + token) *
        kHeadDimension;
    const float first =
        __bfloat162float(reconstructed_misses[source_base + frequency]);
    const float second = __bfloat162float(
        reconstructed_misses[
            source_base + kHalfHeadDimension + frequency]);
    const int selected_chunk = selected_chunk_ids[plan_offset];
    const int64_t position =
        static_cast<int64_t>(selected_chunk) * kChunkSize + token;
    const int64_t frequency_offset =
        position * kHalfHeadDimension + frequency;
    const float cosine_value = cosine[frequency_offset];
    const float sine_value = sine[frequency_offset];
    const int64_t destination_base =
        (static_cast<int64_t>(head) * kSelectedCapacity * kChunkSize +
         selected * kChunkSize + token) *
        kHeadDimension;
    destination_key_values[destination_base + frequency] =
        __float2bfloat16_rn(
            __fadd_rn(
                __fmul_rn(first, cosine_value),
                __fmul_rn(-second, sine_value)));
    destination_key_values[
        destination_base + kHalfHeadDimension + frequency] =
        __float2bfloat16_rn(
            __fadd_rn(
                __fmul_rn(second, cosine_value),
                __fmul_rn(first, sine_value)));
  }
}

__global__ void shadowkv_finalize_key_bmm_a100_kernel(
    const float* __restrict__ cosine,
    const float* __restrict__ sine,
    const int8_t* __restrict__ component_kinds,
    const int32_t* __restrict__ source_slots,
    const int32_t* __restrict__ selected_chunk_ids,
    const int32_t* __restrict__ selected_lengths,
    const int32_t* __restrict__ plan_slots,
    const int32_t* __restrict__ planner_error_codes,
    const uint4* __restrict__ temporal_key_values,
    int rope_rows,
    int64_t temporal_chunks_per_component,
    int plan_capacity,
    uint4* __restrict__ destination_key_values) {
  const int entry = blockIdx.x;
  const int selected = entry % kSelectedCapacity;
  const int head = entry / kSelectedCapacity;
  const int64_t plan_offset =
      static_cast<int64_t>(head) * kSelectedCapacity + selected;
  const int8_t kind = component_kinds[plan_offset];
  const int source_slot = source_slots[plan_offset];
  const int selected_chunk = selected_chunk_ids[plan_offset];
  const int selected_length = selected_lengths[head];
  const int plan_slot = plan_slots[head];
  const bool plan_row_valid =
      planner_error_codes[head] == 0 && plan_slot >= 0 &&
      plan_slot < plan_capacity && selected_length >= 0 &&
      selected_length <= kSelectedCapacity;
  const bool active_ordinal = selected < selected_length;
  const bool hit_valid =
      plan_row_valid && active_ordinal && kind == kHit && source_slot >= 0 &&
      source_slot < temporal_chunks_per_component;
  const bool miss_valid =
      plan_row_valid && active_ordinal && kind == kMiss &&
      selected_chunk >= 0 &&
      (static_cast<int64_t>(selected_chunk) + 1) * kChunkSize <= rope_rows;
  const int64_t destination_vector = plan_offset * kVectorsPerChunk;

  if (hit_valid) {
    for (int vector = threadIdx.x; vector < kVectorsPerChunk;
         vector += blockDim.x) {
      destination_key_values[destination_vector + vector] =
          temporal_key_values[
              static_cast<int64_t>(source_slot) * kVectorsPerChunk + vector];
    }
    return;
  }
  if (!miss_valid) {
    for (int vector = threadIdx.x; vector < kVectorsPerChunk;
         vector += blockDim.x) {
      destination_key_values[destination_vector + vector] = uint4{0, 0, 0, 0};
    }
    return;
  }

  __shared__ __align__(32) __nv_bfloat16 reconstructed[kChunkSize][kHeadDimension];
  __nv_bfloat16* destination_elements =
      reinterpret_cast<__nv_bfloat16*>(destination_key_values) +
      plan_offset * kChunkElements;
  __nv_bfloat16* reconstructed_elements =
      reinterpret_cast<__nv_bfloat16*>(reconstructed);
  for (int index = threadIdx.x; index < kChunkElements; index += blockDim.x) {
    reconstructed_elements[index] = destination_elements[index];
  }
  __syncthreads();
  const int dimension = threadIdx.x;
  if (dimension >= kHeadDimension) {
    return;
  }
  const int paired_dimension =
      dimension < kHalfHeadDimension
      ? dimension + kHalfHeadDimension
      : dimension - kHalfHeadDimension;
  const int frequency = dimension % kHalfHeadDimension;
#pragma unroll
  for (int token = 0; token < kChunkSize; ++token) {
    const int64_t position =
        static_cast<int64_t>(selected_chunk) * kChunkSize + token;
    const float value = __bfloat162float(reconstructed[token][dimension]);
    const float paired =
        __bfloat162float(reconstructed[token][paired_dimension]);
    const float rotated_half =
        dimension < kHalfHeadDimension ? -paired : paired;
    const int64_t frequency_offset =
        position * kHalfHeadDimension + frequency;
    const float direct_product =
        __fmul_rn(value, cosine[frequency_offset]);
    const float rotated_product =
        __fmul_rn(rotated_half, sine[frequency_offset]);
    destination_elements[token * kHeadDimension + dimension] =
        __float2bfloat16_rn(
            __fadd_rn(direct_product, rotated_product));
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
  if (threadIdx.x >= kVectorsPerChunk) {
    return;
  }
  for (int entry = blockIdx.x; entry < kValueEntries; entry += gridDim.x) {
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
          value_miss_chunk_ids[static_cast<int64_t>(head) *
                                   kSelectedCapacity +
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
      continue;
    }
    if (hit_valid) {
      destination_key_values[destination_vector] =
          temporal_key_values[
              static_cast<int64_t>(source_slot) * kVectorsPerChunk +
              threadIdx.x];
      continue;
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
}

PlanGeometry validate_fused_key_a100(
    const at::Tensor& u,
    const at::Tensor& sv,
    const at::Tensor& gathered_u,
    const at::Tensor& cosine,
    const at::Tensor& sine,
    const at::Tensor& component_kinds,
    const at::Tensor& source_slots,
    const at::Tensor& destination_slots,
    const at::Tensor& miss_ordinals,
    const at::Tensor& selected_chunk_ids,
    const at::Tensor& selected_lengths,
    const at::Tensor& plan_slots,
    const at::Tensor& planner_error_codes,
    const at::Tensor& temporal_key_values,
    int64_t plan_capacity,
    const at::Tensor& destination_key_values) {
  check_tensor(u, "u", at::ScalarType::BFloat16, 2);
  check_tensor(sv, "sv", at::ScalarType::BFloat16, 3);
  check_tensor(gathered_u, "gathered_u", at::ScalarType::BFloat16, 3);
  check_tensor(cosine, "cosine", at::ScalarType::Float, 2);
  check_tensor(sine, "sine", at::ScalarType::Float, 2);
  check_tensor(miss_ordinals, "miss_ordinals", at::ScalarType::Int, 3);
  check_tensor(selected_chunk_ids, "selected_chunk_ids", at::ScalarType::Int, 2);
  check_alignment(u, "u", kVectorBytes);
  check_alignment(sv, "sv", kVectorBytes);
  check_alignment(gathered_u, "gathered_u", kVectorBytes);
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
      gathered_u.sizes() ==
          at::IntArrayRef(
              {kKVHeads, kSelectedCapacity * kChunkSize, kRank}),
      "gathered_u must have shape [8, 2048, 160]");
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
      &gathered_u,
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
  return geometry;
}

void launch_prepare_key_a100(
    const at::Tensor& u,
    at::Tensor& gathered_u,
    const at::Tensor& cosine,
    const at::Tensor& component_kinds,
    const at::Tensor& source_slots,
    const at::Tensor& destination_slots,
    const at::Tensor& miss_ordinals,
    const at::Tensor& selected_chunk_ids,
    const at::Tensor& selected_lengths,
    const at::Tensor& plan_slots,
    at::Tensor& planner_error_codes,
    const PlanGeometry& geometry,
    int64_t plan_capacity,
    cudaStream_t stream) {
  TORCH_INTERNAL_ASSERT(stream == at::cuda::getCurrentCUDAStream());
  shadowkv_prepare_key_bmm_a100_kernel
      <<<kKVHeads * kSelectedCapacity, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(u.data_ptr<at::BFloat16>()),
      component_kinds.data_ptr<int8_t>(),
      source_slots.data_ptr<int32_t>(),
      destination_slots.data_ptr<int32_t>(),
      miss_ordinals.data_ptr<int32_t>(),
      selected_chunk_ids.data_ptr<int32_t>(),
      selected_lengths.data_ptr<int32_t>(),
      plan_slots.data_ptr<int32_t>(),
      planner_error_codes.data_ptr<int32_t>(),
      static_cast<int>(u.size(0)),
      static_cast<int>(cosine.size(0)),
      geometry.temporal_chunks_per_component,
      static_cast<int>(plan_capacity),
      reinterpret_cast<__nv_bfloat16*>(
          gathered_u.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_validate_key_miss_plan_a100(
    const at::Tensor& u,
    const at::Tensor& cosine,
    const at::Tensor& component_kinds,
    const at::Tensor& source_slots,
    const at::Tensor& destination_slots,
    const at::Tensor& miss_ordinals,
    const at::Tensor& selected_chunk_ids,
    const at::Tensor& selected_lengths,
    const at::Tensor& plan_slots,
    at::Tensor& planner_error_codes,
    const PlanGeometry& geometry,
    int64_t plan_capacity,
    cudaStream_t stream,
    int32_t* mapped_miss_counts = nullptr) {
  TORCH_INTERNAL_ASSERT(stream == at::cuda::getCurrentCUDAStream());
  shadowkv_validate_key_miss_plan_a100_kernel
      <<<kKVHeads, kThreads, 0, stream>>>(
      component_kinds.data_ptr<int8_t>(),
      source_slots.data_ptr<int32_t>(),
      destination_slots.data_ptr<int32_t>(),
      miss_ordinals.data_ptr<int32_t>(),
      selected_chunk_ids.data_ptr<int32_t>(),
      selected_lengths.data_ptr<int32_t>(),
      plan_slots.data_ptr<int32_t>(),
      planner_error_codes.data_ptr<int32_t>(),
      static_cast<int>(u.size(0)),
      static_cast<int>(cosine.size(0)),
      geometry.temporal_chunks_per_component,
      static_cast<int>(plan_capacity),
      mapped_miss_counts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_materialize_key_misses_a100(
    const at::Tensor& u,
    at::Tensor& gathered_u,
    const at::Tensor& component_kinds,
    const at::Tensor& source_slots,
    const at::Tensor& miss_ordinals,
    const at::Tensor& selected_chunk_ids,
    const at::Tensor& planner_error_codes,
    const at::Tensor& temporal_key_values,
    at::Tensor& destination_key_values,
    cudaStream_t stream) {
  TORCH_INTERNAL_ASSERT(stream == at::cuda::getCurrentCUDAStream());
  shadowkv_materialize_key_misses_a100_kernel
      <<<kKVHeads * kSelectedCapacity, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(u.data_ptr<at::BFloat16>()),
      component_kinds.data_ptr<int8_t>(),
      source_slots.data_ptr<int32_t>(),
      miss_ordinals.data_ptr<int32_t>(),
      selected_chunk_ids.data_ptr<int32_t>(),
      planner_error_codes.data_ptr<int32_t>(),
      reinterpret_cast<const uint4*>(
          temporal_key_values.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(
          gathered_u.data_ptr<at::BFloat16>()),
      reinterpret_cast<uint4*>(
          destination_key_values.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_prepare_key_miss_only_a100(
    const at::Tensor& u,
    at::Tensor& gathered_u,
    const at::Tensor& cosine,
    const at::Tensor& component_kinds,
    const at::Tensor& source_slots,
    const at::Tensor& destination_slots,
    const at::Tensor& miss_ordinals,
    const at::Tensor& selected_chunk_ids,
    const at::Tensor& selected_lengths,
    const at::Tensor& plan_slots,
    at::Tensor& planner_error_codes,
    const at::Tensor& temporal_key_values,
    const PlanGeometry& geometry,
    int64_t plan_capacity,
    at::Tensor& destination_key_values,
    cudaStream_t stream,
    int32_t* mapped_miss_counts = nullptr) {
  launch_validate_key_miss_plan_a100(
      u,
      cosine,
      component_kinds,
      source_slots,
      destination_slots,
      miss_ordinals,
      selected_chunk_ids,
      selected_lengths,
      plan_slots,
      planner_error_codes,
      geometry,
      plan_capacity,
      stream,
      mapped_miss_counts);
  launch_materialize_key_misses_a100(
      u,
      gathered_u,
      component_kinds,
      source_slots,
      miss_ordinals,
      selected_chunk_ids,
      planner_error_codes,
      temporal_key_values,
      destination_key_values,
      stream);
}

void launch_reconstruct_key_misses_a100(
    const at::Tensor& sv,
    const at::Tensor& gathered_u,
    const at::Tensor& cosine,
    const at::Tensor& sine,
    const at::Tensor& component_kinds,
    const at::Tensor& miss_ordinals,
    const at::Tensor& selected_chunk_ids,
    at::Tensor& planner_error_codes,
    at::Tensor& destination_key_values,
    cudaStream_t stream) {
  TORCH_INTERNAL_ASSERT(stream == at::cuda::getCurrentCUDAStream());
  static const bool shared_memory_configured = []() {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        shadowkv_reconstruct_key_misses_a100_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kMissGemmSharedBytes)));
    return true;
  }();
  TORCH_INTERNAL_ASSERT(shared_memory_configured);
  const dim3 grid(kKVHeads, kKeyBlocksPerHead);
  shadowkv_reconstruct_key_misses_a100_kernel
      <<<grid, kMissGemmThreads, kMissGemmSharedBytes, stream>>>(
          reinterpret_cast<const ShadowKVCutlassElement*>(
              gathered_u.data_ptr<at::BFloat16>()),
          reinterpret_cast<const ShadowKVCutlassElement*>(
              sv.data_ptr<at::BFloat16>()),
          cosine.data_ptr<float>(),
          sine.data_ptr<float>(),
          component_kinds.data_ptr<int8_t>(),
          miss_ordinals.data_ptr<int32_t>(),
          selected_chunk_ids.data_ptr<int32_t>(),
          planner_error_codes.data_ptr<int32_t>(),
          reinterpret_cast<ShadowKVCutlassElement*>(
              destination_key_values.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_key_bmm_a100(
    const at::Tensor& sv,
    const at::Tensor& gathered_u,
    at::Tensor& destination_key_values,
    cudaStream_t stream) {
  TORCH_INTERNAL_ASSERT(stream == at::cuda::getCurrentCUDAStream());
  at::Tensor key_destination = destination_key_values.select(0, 0).view(
      {kKVHeads, kSelectedCapacity * kChunkSize, kHeadDimension});
  at::bmm_out(key_destination, gathered_u, sv);
}

void launch_finalize_key_a100(
    const at::Tensor& cosine,
    const at::Tensor& sine,
    const at::Tensor& component_kinds,
    const at::Tensor& source_slots,
    const at::Tensor& selected_chunk_ids,
    const at::Tensor& selected_lengths,
    const at::Tensor& plan_slots,
    at::Tensor& planner_error_codes,
    const at::Tensor& temporal_key_values,
    const PlanGeometry& geometry,
    int64_t plan_capacity,
    at::Tensor& destination_key_values,
    cudaStream_t stream) {
  TORCH_INTERNAL_ASSERT(stream == at::cuda::getCurrentCUDAStream());
  shadowkv_finalize_key_bmm_a100_kernel
      <<<kKVHeads * kSelectedCapacity, kHeadDimension, 0, stream>>>(
          cosine.data_ptr<float>(),
          sine.data_ptr<float>(),
          component_kinds.data_ptr<int8_t>(),
          source_slots.data_ptr<int32_t>(),
          selected_chunk_ids.data_ptr<int32_t>(),
          selected_lengths.data_ptr<int32_t>(),
          plan_slots.data_ptr<int32_t>(),
          planner_error_codes.data_ptr<int32_t>(),
          reinterpret_cast<const uint4*>(
              temporal_key_values.data_ptr<at::BFloat16>()),
          static_cast<int>(cosine.size(0)),
          geometry.temporal_chunks_per_component,
          static_cast<int>(plan_capacity),
          reinterpret_cast<uint4*>(
              destination_key_values.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_fused_key_a100(
    const at::Tensor& u,
    const at::Tensor& sv,
    at::Tensor& gathered_u,
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
    const PlanGeometry& geometry,
    int64_t plan_capacity,
    at::Tensor& destination_key_values,
    cudaStream_t stream) {
  launch_prepare_key_a100(
      u,
      gathered_u,
      cosine,
      component_kinds,
      source_slots,
      destination_slots,
      miss_ordinals,
      selected_chunk_ids,
      selected_lengths,
      plan_slots,
      planner_error_codes,
      geometry,
      plan_capacity,
      stream);
  launch_key_bmm_a100(sv, gathered_u, destination_key_values, stream);
  launch_finalize_key_a100(
      cosine,
      sine,
      component_kinds,
      source_slots,
      selected_chunk_ids,
      selected_lengths,
      plan_slots,
      planner_error_codes,
      temporal_key_values,
      geometry,
      plan_capacity,
      destination_key_values,
      stream);
}

void launch_fused_key_miss_only_a100(
    const at::Tensor& u,
    const at::Tensor& sv,
    at::Tensor& gathered_u,
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
    const PlanGeometry& geometry,
    int64_t plan_capacity,
    at::Tensor& destination_key_values,
    cudaStream_t stream) {
  launch_prepare_key_miss_only_a100(
      u,
      gathered_u,
      cosine,
      component_kinds,
      source_slots,
      destination_slots,
      miss_ordinals,
      selected_chunk_ids,
      selected_lengths,
      plan_slots,
      planner_error_codes,
      temporal_key_values,
      geometry,
      plan_capacity,
      destination_key_values,
      stream);
  launch_reconstruct_key_misses_a100(
      sv,
      gathered_u,
      cosine,
      sine,
      component_kinds,
      miss_ordinals,
      selected_chunk_ids,
      planner_error_codes,
      destination_key_values,
      stream);
}

void validate_exact_miss_resources(
    const at::Tensor& reconstructed_misses,
    const at::Tensor& host_miss_counts,
    int64_t mapped_miss_counts,
    int64_t miss_count_ready_event,
    const at::Device& device) {
  check_tensor(
      reconstructed_misses,
      "reconstructed_misses",
      at::ScalarType::BFloat16,
      3);
  TORCH_CHECK(
      reconstructed_misses.sizes() ==
          at::IntArrayRef(
              {kKVHeads, kSelectedCapacity * kChunkSize, kHeadDimension}),
      "reconstructed_misses must have shape [8, 2048, 128]");
  TORCH_CHECK(
      reconstructed_misses.device() == device,
      "exact miss workspace must share the fused-key CUDA device");
  check_alignment(reconstructed_misses, "reconstructed_misses", kVectorBytes);
  TORCH_CHECK(
      host_miss_counts.device().is_cpu() &&
          host_miss_counts.is_contiguous() &&
          host_miss_counts.scalar_type() == at::ScalarType::Int &&
          host_miss_counts.sizes() == at::IntArrayRef({kKVHeads}) &&
          host_miss_counts.is_pinned(),
      "host_miss_counts must be pinned contiguous int32 [8]");
  TORCH_CHECK(
      mapped_miss_counts > 0 && mapped_miss_counts % alignof(int32_t) == 0,
      "mapped miss-count pointer is invalid");
  TORCH_CHECK(
      miss_count_ready_event > 0,
      "miss-count readiness event is invalid");
  ExactMissGemmRuntime& runtime = exact_miss_gemm_runtime();
  TORCH_CHECK(
      runtime.prepared && runtime.handle != nullptr &&
          runtime.device_index == device.index(),
      "exact A100 miss GEMM was not prepared for this CUDA device");
}

void record_exact_miss_counts_ready(
    int64_t miss_count_ready_event,
    cudaStream_t stream) {
  cudaEvent_t ready_event =
      reinterpret_cast<cudaEvent_t>(miss_count_ready_event);
  C10_CUDA_CHECK(cudaEventRecord(ready_event, stream));
}

MissCounts await_exact_miss_counts(
    at::Tensor& host_miss_counts,
    int64_t miss_count_ready_event) {
  auto* host_counts = host_miss_counts.data_ptr<int32_t>();
  cudaEvent_t ready_event =
      reinterpret_cast<cudaEvent_t>(miss_count_ready_event);
  C10_CUDA_CHECK(cudaEventSynchronize(ready_event));
  MissCounts counts{};
  for (int head = 0; head < kKVHeads; ++head) {
    TORCH_CHECK(
        host_counts[head] >= 0 && host_counts[head] <= kSelectedCapacity,
        "exact A100 miss count was not published safely");
    counts.values[head] = host_counts[head];
  }
  return counts;
}

void reset_exact_miss_counts(at::Tensor& host_miss_counts) {
  auto* host_counts = host_miss_counts.data_ptr<int32_t>();
  std::fill(host_counts, host_counts + kKVHeads, -1);
}

void launch_exact_miss_gemm(
    const at::Tensor& gathered_u,
    const at::Tensor& sv,
    at::Tensor& reconstructed_misses,
    MissCounts miss_counts,
    cudaStream_t stream) {
  int maximum_miss_chunks = 0;
  for (int head = 0; head < kKVHeads; ++head) {
    maximum_miss_chunks =
        std::max(maximum_miss_chunks, miss_counts.values[head]);
  }
  if (maximum_miss_chunks == 0) {
    return;
  }
  const int rows = maximum_miss_chunks * kChunkSize;
  shadowkv_zero_compact_key_padding_a100_kernel
      <<<dim3(kKVHeads, 8), kThreads, 0, stream>>>(
          reinterpret_cast<__nv_bfloat16*>(
              gathered_u.data_ptr<at::BFloat16>()),
          miss_counts,
          maximum_miss_chunks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  ExactMissGemmRuntime& runtime = exact_miss_gemm_runtime();
  ExactMatrixLayouts layouts(rows);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_cublas(
      cublasLtMatmul(
          runtime.handle,
          &runtime.operation,
          &alpha,
          gathered_u.data_ptr(),
          &layouts.a,
          sv.data_ptr(),
          &layouts.b,
          &beta,
          reconstructed_misses.data_ptr(),
          &layouts.c,
          reconstructed_misses.data_ptr(),
          &layouts.d,
          &runtime.algorithm,
          nullptr,
          0,
          stream),
      "launch exact compact A100 miss GEMM");
}

void launch_finalize_exact_misses(
    const at::Tensor& reconstructed_misses,
    const at::Tensor& cosine,
    const at::Tensor& sine,
    const at::Tensor& component_kinds,
    const at::Tensor& miss_ordinals,
    const at::Tensor& selected_chunk_ids,
    at::Tensor& destination_key_values,
    MissCounts miss_counts,
    cudaStream_t stream) {
  const bool has_misses = std::any_of(
      std::begin(miss_counts.values),
      std::end(miss_counts.values),
      [](int32_t count) { return count > 0; });
  if (!has_misses) {
    return;
  }
  shadowkv_finalize_compact_key_misses_a100_kernel
      <<<dim3(kKVHeads, kKeyBlocksPerHead), kThreads, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(
              reconstructed_misses.data_ptr<at::BFloat16>()),
          cosine.data_ptr<float>(),
          sine.data_ptr<float>(),
          component_kinds.data_ptr<int8_t>(),
          miss_ordinals.data_ptr<int32_t>(),
          selected_chunk_ids.data_ptr<int32_t>(),
          reinterpret_cast<__nv_bfloat16*>(
              destination_key_values.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void finish_exact_fused_key_a100(
    const at::Tensor& sv,
    const at::Tensor& gathered_u,
    at::Tensor& reconstructed_misses,
    const at::Tensor& cosine,
    const at::Tensor& sine,
    const at::Tensor& component_kinds,
    const at::Tensor& miss_ordinals,
    const at::Tensor& selected_chunk_ids,
    at::Tensor& host_miss_counts,
    int64_t miss_count_ready_event,
    at::Tensor& destination_key_values,
    cudaStream_t stream) {
  const MissCounts counts = await_exact_miss_counts(
      host_miss_counts, miss_count_ready_event);
  launch_exact_miss_gemm(
      gathered_u,
      sv,
      reconstructed_misses,
      counts,
      stream);
  launch_finalize_exact_misses(
      reconstructed_misses,
      cosine,
      sine,
      component_kinds,
      miss_ordinals,
      selected_chunk_ids,
      destination_key_values,
      counts,
      stream);
}

PlanGeometry validate_mapped_value_a100(
    const at::Tensor& component_kinds,
    const at::Tensor& source_slots,
    const at::Tensor& destination_slots,
    const at::Tensor& miss_ordinals,
    const at::Tensor& selected_chunk_ids,
    const at::Tensor& selected_lengths,
    const at::Tensor& plan_slots,
    const at::Tensor& planner_error_codes,
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
    const at::Tensor& destination_key_values) {
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
  return geometry;
}

void launch_mapped_value_a100(
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
    int64_t prompt_chunk_capacity,
    int64_t prompt_tokens,
    int64_t expected_generation,
    const PlanGeometry& geometry,
    int64_t plan_capacity,
    at::Tensor& destination_key_values,
    cudaStream_t stream,
    int grid_blocks = kValueEntries) {
  shadowkv_place_value_a100_kernel<ValueSource::kMappedHost>
      <<<grid_blocks, kVectorsPerChunk, 0, stream>>>(
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

}  // namespace

void shadowkv_prepare_exact_miss_gemm_a100(const at::Tensor& device_anchor) {
  TORCH_CHECK(device_anchor.is_cuda(), "exact miss GEMM anchor must use CUDA");
  c10::cuda::CUDAGuard device_guard(device_anchor.device());
  check_sm80(device_anchor);
  prepare_exact_miss_gemm(device_anchor.get_device());
}

int64_t shadowkv_resolve_miss_count_pointer_a100(
    const at::Tensor& host_miss_counts,
    int64_t device_index) {
  TORCH_CHECK(
      host_miss_counts.device().is_cpu() &&
          host_miss_counts.is_contiguous() &&
          host_miss_counts.scalar_type() == at::ScalarType::Int &&
          host_miss_counts.sizes() == at::IntArrayRef({kKVHeads}) &&
          host_miss_counts.is_pinned(),
      "host miss counts must be pinned contiguous int32 [8]");
  TORCH_CHECK(
      device_index >= 0 && device_index <= std::numeric_limits<int>::max(),
      "mapped miss-count CUDA device is invalid");
  int device_count = 0;
  C10_CUDA_CHECK(cudaGetDeviceCount(&device_count));
  TORCH_CHECK(
      device_index < device_count,
      "mapped miss-count CUDA device is not visible");
  c10::cuda::CUDAGuard device_guard(
      c10::Device(c10::DeviceType::CUDA, device_index));
  check_sm80_device(static_cast<int>(device_index));
  void* device_pointer = nullptr;
  C10_CUDA_CHECK(cudaHostGetDevicePointer(
      &device_pointer,
      host_miss_counts.data_ptr<int32_t>(),
      0));
  TORCH_CHECK(
      device_pointer != nullptr,
      "pinned host miss counts have no mapped CUDA pointer");
  const uintptr_t pointer = reinterpret_cast<uintptr_t>(device_pointer);
  TORCH_CHECK(
      pointer % alignof(int32_t) == 0 &&
          pointer <= static_cast<uintptr_t>(std::numeric_limits<int64_t>::max()),
      "mapped miss-count pointer exceeds the operator contract");
  return static_cast<int64_t>(pointer);
}

void shadowkv_fused_key_a100(
    const at::Tensor& u,
    const at::Tensor& sv,
    at::Tensor& gathered_u,
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
  const PlanGeometry geometry = validate_fused_key_a100(
          u,
          sv,
          gathered_u,
          cosine,
          sine,
          component_kinds,
          source_slots,
          destination_slots,
          miss_ordinals,
          selected_chunk_ids,
          selected_lengths,
          plan_slots,
          planner_error_codes,
          temporal_key_values,
          plan_capacity,
          destination_key_values);
  const auto device = u.device();
  c10::cuda::CUDAGuard device_guard(device);
  check_sm80(u);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_fused_key_a100(
      u,
      sv,
      gathered_u,
      cosine,
      sine,
      component_kinds,
      source_slots,
      destination_slots,
      miss_ordinals,
      selected_chunk_ids,
      selected_lengths,
      plan_slots,
      planner_error_codes,
      temporal_key_values,
      geometry,
      plan_capacity,
      destination_key_values,
      stream);
}

void shadowkv_fused_key_miss_only_a100(
    const at::Tensor& u,
    const at::Tensor& sv,
    at::Tensor& gathered_u,
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
  const PlanGeometry geometry = validate_fused_key_a100(
      u,
      sv,
      gathered_u,
      cosine,
      sine,
      component_kinds,
      source_slots,
      destination_slots,
      miss_ordinals,
      selected_chunk_ids,
      selected_lengths,
      plan_slots,
      planner_error_codes,
      temporal_key_values,
      plan_capacity,
      destination_key_values);
  const auto device = u.device();
  c10::cuda::CUDAGuard device_guard(device);
  check_sm80(u);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_fused_key_miss_only_a100(
      u,
      sv,
      gathered_u,
      cosine,
      sine,
      component_kinds,
      source_slots,
      destination_slots,
      miss_ordinals,
      selected_chunk_ids,
      selected_lengths,
      plan_slots,
      planner_error_codes,
      temporal_key_values,
      geometry,
      plan_capacity,
      destination_key_values,
      stream);
}

void shadowkv_fused_key_exact_a100(
    const at::Tensor& u,
    const at::Tensor& sv,
    at::Tensor& gathered_u,
    at::Tensor& reconstructed_misses,
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
    at::Tensor& host_miss_counts,
    int64_t mapped_miss_counts,
    int64_t miss_count_ready_event,
    int64_t plan_capacity,
    at::Tensor& destination_key_values) {
  const PlanGeometry geometry = validate_fused_key_a100(
      u,
      sv,
      gathered_u,
      cosine,
      sine,
      component_kinds,
      source_slots,
      destination_slots,
      miss_ordinals,
      selected_chunk_ids,
      selected_lengths,
      plan_slots,
      planner_error_codes,
      temporal_key_values,
      plan_capacity,
      destination_key_values);
  const auto device = u.device();
  validate_exact_miss_resources(
      reconstructed_misses,
      host_miss_counts,
      mapped_miss_counts,
      miss_count_ready_event,
      device);
  c10::cuda::CUDAGuard device_guard(device);
  check_sm80(u);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  reset_exact_miss_counts(host_miss_counts);
  launch_validate_key_miss_plan_a100(
      u,
      cosine,
      component_kinds,
      source_slots,
      destination_slots,
      miss_ordinals,
      selected_chunk_ids,
      selected_lengths,
      plan_slots,
      planner_error_codes,
      geometry,
      plan_capacity,
      stream,
      reinterpret_cast<int32_t*>(
          static_cast<uintptr_t>(mapped_miss_counts)));
  record_exact_miss_counts_ready(miss_count_ready_event, stream);
  launch_materialize_key_misses_a100(
      u,
      gathered_u,
      component_kinds,
      source_slots,
      miss_ordinals,
      selected_chunk_ids,
      planner_error_codes,
      temporal_key_values,
      destination_key_values,
      stream);
  finish_exact_fused_key_a100(
      sv,
      gathered_u,
      reconstructed_misses,
      cosine,
      sine,
      component_kinds,
      miss_ordinals,
      selected_chunk_ids,
      host_miss_counts,
      miss_count_ready_event,
      destination_key_values,
      stream);
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
      <<<kKVHeads * kSelectedCapacity, kVectorsPerChunk, 0, stream>>>(
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
      <<<kKVHeads * kSelectedCapacity, kVectorsPerChunk, 0, stream>>>(
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
  const PlanGeometry geometry = validate_mapped_value_a100(
      component_kinds,
      source_slots,
      destination_slots,
      miss_ordinals,
      selected_chunk_ids,
      selected_lengths,
      plan_slots,
      planner_error_codes,
      temporal_key_values,
      value_miss_chunk_ids,
      value_miss_lengths,
      descriptor_generation,
      descriptor_validity,
      mapped_host_pointer,
      mapped_host_bytes,
      prompt_chunk_capacity,
      prompt_tokens,
      expected_generation,
      plan_capacity,
      destination_key_values);
  const auto device = component_kinds.device();
  c10::cuda::CUDAGuard device_guard(device);
  check_sm80(component_kinds);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_mapped_value_a100(
      component_kinds,
      source_slots,
      destination_slots,
      miss_ordinals,
      selected_chunk_ids,
      selected_lengths,
      plan_slots,
      planner_error_codes,
      temporal_key_values,
      value_miss_chunk_ids,
      value_miss_lengths,
      descriptor_generation,
      descriptor_validity,
      mapped_host_pointer,
      prompt_chunk_capacity,
      prompt_tokens,
      expected_generation,
      geometry,
      plan_capacity,
      destination_key_values,
      stream);
}

void shadowkv_fused_key_mapped_value_a100(
    const at::Tensor& u,
    const at::Tensor& sv,
    at::Tensor& gathered_u,
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
    int64_t reconstruction_stream,
    at::Tensor& destination_key_values) {
  const PlanGeometry key_geometry = validate_fused_key_a100(
      u,
      sv,
      gathered_u,
      cosine,
      sine,
      component_kinds,
      source_slots,
      destination_slots,
      miss_ordinals,
      selected_chunk_ids,
      selected_lengths,
      plan_slots,
      planner_error_codes,
      temporal_key_values,
      plan_capacity,
      destination_key_values);
  const PlanGeometry value_geometry = validate_mapped_value_a100(
      component_kinds,
      source_slots,
      destination_slots,
      miss_ordinals,
      selected_chunk_ids,
      selected_lengths,
      plan_slots,
      planner_error_codes,
      temporal_key_values,
      value_miss_chunk_ids,
      value_miss_lengths,
      descriptor_generation,
      descriptor_validity,
      mapped_host_pointer,
      mapped_host_bytes,
      prompt_chunk_capacity,
      prompt_tokens,
      expected_generation,
      plan_capacity,
      destination_key_values);
  const auto device = component_kinds.device();
  c10::cuda::CUDAGuard device_guard(device);
  check_sm80(component_kinds);
  cudaStream_t value_stream = at::cuda::getCurrentCUDAStream();
  cudaStream_t key_stream = reinterpret_cast<cudaStream_t>(reconstruction_stream);
  int key_stream_device = -1;
  C10_CUDA_CHECK(cudaStreamGetDevice(key_stream, &key_stream_device));
  TORCH_CHECK(
      key_stream_device == device.index(),
      "A100 fused reconstruction stream must target the plan CUDA device");
  TORCH_CHECK(
      key_stream != value_stream,
      "A100 fused reconstruction and mapped-value streams must be distinct");

  launch_mapped_value_a100(
      component_kinds,
      source_slots,
      destination_slots,
      miss_ordinals,
      selected_chunk_ids,
      selected_lengths,
      plan_slots,
      planner_error_codes,
      temporal_key_values,
      value_miss_chunk_ids,
      value_miss_lengths,
      descriptor_generation,
      descriptor_validity,
      mapped_host_pointer,
      prompt_chunk_capacity,
      prompt_tokens,
      expected_generation,
      value_geometry,
      plan_capacity,
      destination_key_values,
      value_stream);

  const c10::cuda::CUDAStream external_key_stream =
      c10::cuda::getStreamFromExternal(key_stream, device.index());
  c10::cuda::CUDAStreamGuard key_stream_guard(external_key_stream);
  launch_fused_key_a100(
      u,
      sv,
      gathered_u,
      cosine,
      sine,
      component_kinds,
      source_slots,
      destination_slots,
      miss_ordinals,
      selected_chunk_ids,
      selected_lengths,
      plan_slots,
      planner_error_codes,
      temporal_key_values,
      key_geometry,
      plan_capacity,
      destination_key_values,
      key_stream);
}

static void launch_fused_key_mapped_value_staged_a100(
    const at::Tensor& u,
    const at::Tensor& sv,
    at::Tensor& gathered_u,
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
    int64_t reconstruction_stream,
    at::Tensor& destination_key_values,
    int mapped_value_grid_blocks,
    bool miss_only_key) {
  const PlanGeometry key_geometry = validate_fused_key_a100(
      u,
      sv,
      gathered_u,
      cosine,
      sine,
      component_kinds,
      source_slots,
      destination_slots,
      miss_ordinals,
      selected_chunk_ids,
      selected_lengths,
      plan_slots,
      planner_error_codes,
      temporal_key_values,
      plan_capacity,
      destination_key_values);
  const PlanGeometry value_geometry = validate_mapped_value_a100(
      component_kinds,
      source_slots,
      destination_slots,
      miss_ordinals,
      selected_chunk_ids,
      selected_lengths,
      plan_slots,
      planner_error_codes,
      temporal_key_values,
      value_miss_chunk_ids,
      value_miss_lengths,
      descriptor_generation,
      descriptor_validity,
      mapped_host_pointer,
      mapped_host_bytes,
      prompt_chunk_capacity,
      prompt_tokens,
      expected_generation,
      plan_capacity,
      destination_key_values);
  const auto device = component_kinds.device();
  c10::cuda::CUDAGuard device_guard(device);
  check_sm80(component_kinds);
  cudaStream_t value_stream = at::cuda::getCurrentCUDAStream();
  cudaStream_t key_stream = reinterpret_cast<cudaStream_t>(reconstruction_stream);
  int key_stream_device = -1;
  C10_CUDA_CHECK(cudaStreamGetDevice(key_stream, &key_stream_device));
  TORCH_CHECK(
      key_stream_device == device.index(),
      "A100 fused reconstruction stream must target the plan CUDA device");
  TORCH_CHECK(
      key_stream != value_stream,
      "A100 fused reconstruction and mapped-value streams must be distinct");

  const c10::cuda::CUDAStream external_key_stream =
      c10::cuda::getStreamFromExternal(key_stream, device.index());
  {
    c10::cuda::CUDAStreamGuard key_stream_guard(external_key_stream);
    if (miss_only_key) {
      launch_prepare_key_miss_only_a100(
          u,
          gathered_u,
          cosine,
          component_kinds,
          source_slots,
          destination_slots,
          miss_ordinals,
          selected_chunk_ids,
          selected_lengths,
          plan_slots,
          planner_error_codes,
          temporal_key_values,
          key_geometry,
          plan_capacity,
          destination_key_values,
          key_stream);
    } else {
      launch_prepare_key_a100(
          u,
          gathered_u,
          cosine,
          component_kinds,
          source_slots,
          destination_slots,
          miss_ordinals,
          selected_chunk_ids,
          selected_lengths,
          plan_slots,
          planner_error_codes,
          key_geometry,
          plan_capacity,
          key_stream);
    }
  }

  launch_mapped_value_a100(
      component_kinds,
      source_slots,
      destination_slots,
      miss_ordinals,
      selected_chunk_ids,
      selected_lengths,
      plan_slots,
      planner_error_codes,
      temporal_key_values,
      value_miss_chunk_ids,
      value_miss_lengths,
      descriptor_generation,
      descriptor_validity,
      mapped_host_pointer,
      prompt_chunk_capacity,
      prompt_tokens,
      expected_generation,
      value_geometry,
      plan_capacity,
      destination_key_values,
      value_stream,
      mapped_value_grid_blocks);

  {
    c10::cuda::CUDAStreamGuard key_stream_guard(external_key_stream);
    if (miss_only_key) {
      launch_reconstruct_key_misses_a100(
          sv,
          gathered_u,
          cosine,
          sine,
          component_kinds,
          miss_ordinals,
          selected_chunk_ids,
          planner_error_codes,
          destination_key_values,
          key_stream);
    } else {
      launch_key_bmm_a100(sv, gathered_u, destination_key_values, key_stream);
      launch_finalize_key_a100(
          cosine,
          sine,
          component_kinds,
          source_slots,
          selected_chunk_ids,
          selected_lengths,
          plan_slots,
          planner_error_codes,
          temporal_key_values,
          key_geometry,
          plan_capacity,
          destination_key_values,
          key_stream);
    }
  }
}

void shadowkv_fused_key_mapped_value_staged_a100(
    const at::Tensor& u,
    const at::Tensor& sv,
    at::Tensor& gathered_u,
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
    int64_t reconstruction_stream,
    at::Tensor& destination_key_values) {
  launch_fused_key_mapped_value_staged_a100(
      u,
      sv,
      gathered_u,
      cosine,
      sine,
      component_kinds,
      source_slots,
      destination_slots,
      miss_ordinals,
      selected_chunk_ids,
      selected_lengths,
      plan_slots,
      planner_error_codes,
      temporal_key_values,
      value_miss_chunk_ids,
      value_miss_lengths,
      descriptor_generation,
      descriptor_validity,
      mapped_host_pointer,
      mapped_host_bytes,
      prompt_chunk_capacity,
      prompt_tokens,
      expected_generation,
      plan_capacity,
      reconstruction_stream,
      destination_key_values,
      kValueEntries,
      false);
}

void shadowkv_fused_key_mapped_value_throttled_a100(
    const at::Tensor& u,
    const at::Tensor& sv,
    at::Tensor& gathered_u,
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
    int64_t reconstruction_stream,
    at::Tensor& destination_key_values) {
  launch_fused_key_mapped_value_staged_a100(
      u,
      sv,
      gathered_u,
      cosine,
      sine,
      component_kinds,
      source_slots,
      destination_slots,
      miss_ordinals,
      selected_chunk_ids,
      selected_lengths,
      plan_slots,
      planner_error_codes,
      temporal_key_values,
      value_miss_chunk_ids,
      value_miss_lengths,
      descriptor_generation,
      descriptor_validity,
      mapped_host_pointer,
      mapped_host_bytes,
      prompt_chunk_capacity,
      prompt_tokens,
      expected_generation,
      plan_capacity,
      reconstruction_stream,
      destination_key_values,
      kThrottledMappedValueBlocks,
      false);
}

void shadowkv_fused_key_mapped_value_miss_only_a100(
    const at::Tensor& u,
    const at::Tensor& sv,
    at::Tensor& gathered_u,
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
    int64_t reconstruction_stream,
    at::Tensor& destination_key_values) {
  launch_fused_key_mapped_value_staged_a100(
      u,
      sv,
      gathered_u,
      cosine,
      sine,
      component_kinds,
      source_slots,
      destination_slots,
      miss_ordinals,
      selected_chunk_ids,
      selected_lengths,
      plan_slots,
      planner_error_codes,
      temporal_key_values,
      value_miss_chunk_ids,
      value_miss_lengths,
      descriptor_generation,
      descriptor_validity,
      mapped_host_pointer,
      mapped_host_bytes,
      prompt_chunk_capacity,
      prompt_tokens,
      expected_generation,
      plan_capacity,
      reconstruction_stream,
      destination_key_values,
      kThrottledMappedValueBlocks,
      true);
}

void shadowkv_fused_key_mapped_value_exact_a100(
    const at::Tensor& u,
    const at::Tensor& sv,
    at::Tensor& gathered_u,
    at::Tensor& reconstructed_misses,
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
    const at::Tensor& value_miss_chunk_ids,
    const at::Tensor& value_miss_lengths,
    const at::Tensor& descriptor_generation,
    const at::Tensor& descriptor_validity,
    int64_t mapped_host_pointer,
    int64_t mapped_host_bytes,
    int64_t prompt_chunk_capacity,
    int64_t prompt_tokens,
    int64_t expected_generation,
    at::Tensor& host_miss_counts,
    int64_t mapped_miss_counts,
    int64_t miss_count_ready_event,
    int64_t plan_capacity,
    int64_t reconstruction_stream,
    at::Tensor& destination_key_values) {
  const PlanGeometry key_geometry = validate_fused_key_a100(
      u,
      sv,
      gathered_u,
      cosine,
      sine,
      component_kinds,
      source_slots,
      destination_slots,
      miss_ordinals,
      selected_chunk_ids,
      selected_lengths,
      plan_slots,
      planner_error_codes,
      temporal_key_values,
      plan_capacity,
      destination_key_values);
  const PlanGeometry value_geometry = validate_mapped_value_a100(
      component_kinds,
      source_slots,
      destination_slots,
      miss_ordinals,
      selected_chunk_ids,
      selected_lengths,
      plan_slots,
      planner_error_codes,
      temporal_key_values,
      value_miss_chunk_ids,
      value_miss_lengths,
      descriptor_generation,
      descriptor_validity,
      mapped_host_pointer,
      mapped_host_bytes,
      prompt_chunk_capacity,
      prompt_tokens,
      expected_generation,
      plan_capacity,
      destination_key_values);
  const auto device = component_kinds.device();
  validate_exact_miss_resources(
      reconstructed_misses,
      host_miss_counts,
      mapped_miss_counts,
      miss_count_ready_event,
      device);
  c10::cuda::CUDAGuard device_guard(device);
  check_sm80(component_kinds);
  cudaStream_t value_stream = at::cuda::getCurrentCUDAStream();
  cudaStream_t key_stream =
      reinterpret_cast<cudaStream_t>(reconstruction_stream);
  int key_stream_device = -1;
  C10_CUDA_CHECK(cudaStreamGetDevice(key_stream, &key_stream_device));
  TORCH_CHECK(
      key_stream_device == device.index(),
      "A100 fused reconstruction stream must target the plan CUDA device");
  TORCH_CHECK(
      key_stream != value_stream,
      "A100 fused reconstruction and mapped-value streams must be distinct");
  const c10::cuda::CUDAStream external_key_stream =
      c10::cuda::getStreamFromExternal(key_stream, device.index());
  reset_exact_miss_counts(host_miss_counts);
  {
    c10::cuda::CUDAStreamGuard key_stream_guard(external_key_stream);
    launch_validate_key_miss_plan_a100(
        u,
        cosine,
        component_kinds,
        source_slots,
        destination_slots,
        miss_ordinals,
        selected_chunk_ids,
        selected_lengths,
        plan_slots,
        planner_error_codes,
        key_geometry,
        plan_capacity,
        key_stream,
        reinterpret_cast<int32_t*>(
            static_cast<uintptr_t>(mapped_miss_counts)));
    record_exact_miss_counts_ready(miss_count_ready_event, key_stream);
    launch_materialize_key_misses_a100(
        u,
        gathered_u,
        component_kinds,
        source_slots,
        miss_ordinals,
        selected_chunk_ids,
        planner_error_codes,
        temporal_key_values,
        destination_key_values,
        key_stream);
  }
  launch_mapped_value_a100(
      component_kinds,
      source_slots,
      destination_slots,
      miss_ordinals,
      selected_chunk_ids,
      selected_lengths,
      plan_slots,
      planner_error_codes,
      temporal_key_values,
      value_miss_chunk_ids,
      value_miss_lengths,
      descriptor_generation,
      descriptor_validity,
      mapped_host_pointer,
      prompt_chunk_capacity,
      prompt_tokens,
      expected_generation,
      value_geometry,
      plan_capacity,
      destination_key_values,
      value_stream,
      kThrottledMappedValueBlocks);
  {
    c10::cuda::CUDAStreamGuard key_stream_guard(external_key_stream);
    finish_exact_fused_key_a100(
        sv,
        gathered_u,
        reconstructed_misses,
        cosine,
        sine,
        component_kinds,
        miss_ordinals,
        selected_chunk_ids,
        host_miss_counts,
        miss_count_ready_event,
        destination_key_values,
        key_stream);
  }
}
