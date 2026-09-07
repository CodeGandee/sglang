/* Copyright 2026 SGLang Team. All Rights Reserved. */

#include <torch/library.h>

#include "sgl_kernel_ops.h"
#include "shadowkv/common/operations.h"

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def(
      "shadowkv_reconstruct_generic_aot_v1(Tensor u, Tensor sv, Tensor positions, Tensor! output) -> ()");
  m.impl("shadowkv_reconstruct_generic_aot_v1", torch::kCUDA, &shadowkv_reconstruct);
  m.def(
      "shadowkv_reconstruct_rope_generic_aot_v1(Tensor u, Tensor sv, Tensor positions, Tensor inverse_frequencies, "
      "Tensor! output) -> ()");
  m.impl("shadowkv_reconstruct_rope_generic_aot_v1", torch::kCUDA, &shadowkv_reconstruct_rope);
  m.def(
      "shadowkv_plan_reuse_generic_aot_v1(Tensor previous_chunks, Tensor previous_lengths, Tensor current_chunks, "
      "Tensor current_lengths, Tensor exact_chunks, Tensor exact_lengths, Tensor cached_generations, Tensor "
      "current_generations, int max_reuse_chunks, int chunk_size, Tensor! plan, Tensor! "
      "deduplicated_exact_chunks, Tensor! counts, Tensor! error_codes) -> ()");
  m.impl("shadowkv_plan_reuse_generic_aot_v1", torch::kCUDA, &shadowkv_plan_reuse);
  m.def(
      "shadowkv_packed_gqa_generic_aot_v1(Tensor query, Tensor keys, Tensor values, Tensor lengths, Tensor! weights, "
      "Tensor! output) -> ()");
  m.impl("shadowkv_packed_gqa_generic_aot_v1", torch::kCUDA, &shadowkv_packed_gqa);
}

REGISTER_EXTENSION(shadowkv_ops)
