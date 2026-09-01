/* Copyright 2026 SGLang Team. All Rights Reserved. */

#include <torch/library.h>

#include "sgl_kernel_ops.h"
#include "shadowkv/sm80/fused_key.h"

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def(
      "shadowkv_fused_key_sm80_a100_v2(Tensor u, Tensor sv, Tensor! gathered_u, Tensor cosine, Tensor sine, Tensor component_kinds, "
      "Tensor source_slots, Tensor destination_slots, Tensor miss_ordinals, Tensor selected_chunk_ids, Tensor "
      "selected_lengths, Tensor plan_slots, Tensor! planner_error_codes, Tensor temporal_key_values, int "
      "plan_capacity, Tensor! destination_key_values) -> ()");
  m.impl(
      "shadowkv_fused_key_sm80_a100_v2",
      torch::kCUDA,
      &shadowkv_fused_key_a100);
  m.def(
      "shadowkv_place_value_sm80_a100_v1(Tensor component_kinds, Tensor source_slots, Tensor destination_slots, "
      "Tensor selected_lengths, Tensor plan_slots, Tensor! planner_error_codes, Tensor temporal_key_values, "
      "Tensor compatibility_key_values, int plan_capacity, Tensor! destination_key_values) -> ()");
  m.impl(
      "shadowkv_place_value_sm80_a100_v1",
      torch::kCUDA,
      &shadowkv_place_value_a100);
  m.def(
      "shadowkv_place_value_miss_only_sm80_a100_v1(Tensor component_kinds, Tensor source_slots, Tensor "
      "destination_slots, Tensor miss_ordinals, Tensor selected_chunk_ids, Tensor selected_lengths, Tensor "
      "plan_slots, Tensor! planner_error_codes, Tensor temporal_key_values, Tensor value_miss_chunk_ids, Tensor "
      "value_miss_lengths, Tensor descriptor_generation, Tensor descriptor_validity, int expected_generation, int "
      "plan_capacity, Tensor value_miss_key_values, Tensor! destination_key_values) -> ()");
  m.impl(
      "shadowkv_place_value_miss_only_sm80_a100_v1",
      torch::kCUDA,
      &shadowkv_place_value_miss_only_a100);
  m.def(
      "shadowkv_place_value_mapped_host_sm80_a100_v1(Tensor component_kinds, Tensor source_slots, Tensor "
      "destination_slots, Tensor miss_ordinals, Tensor selected_chunk_ids, Tensor selected_lengths, Tensor "
      "plan_slots, Tensor! planner_error_codes, Tensor temporal_key_values, Tensor value_miss_chunk_ids, Tensor "
      "value_miss_lengths, Tensor descriptor_generation, Tensor descriptor_validity, int mapped_host_pointer, int "
      "mapped_host_bytes, int prompt_chunk_capacity, int prompt_tokens, int expected_generation, int plan_capacity, "
      "Tensor! destination_key_values) -> ()");
  m.impl(
      "shadowkv_place_value_mapped_host_sm80_a100_v1",
      torch::kCUDA,
      &shadowkv_place_value_mapped_host_a100);
  m.def(
      "shadowkv_fused_key_mapped_value_sm80_a100_v3(Tensor u, Tensor sv, Tensor! gathered_u, Tensor cosine, "
      "Tensor sine, Tensor component_kinds, Tensor source_slots, Tensor destination_slots, Tensor miss_ordinals, "
      "Tensor selected_chunk_ids, Tensor selected_lengths, Tensor plan_slots, Tensor! planner_error_codes, Tensor "
      "temporal_key_values, Tensor value_miss_chunk_ids, Tensor value_miss_lengths, Tensor descriptor_generation, "
      "Tensor descriptor_validity, int mapped_host_pointer, int mapped_host_bytes, int prompt_chunk_capacity, int "
      "prompt_tokens, int expected_generation, int plan_capacity, int reconstruction_stream, Tensor! "
      "destination_key_values) -> ()");
  m.impl(
      "shadowkv_fused_key_mapped_value_sm80_a100_v3",
      torch::kCUDA,
      &shadowkv_fused_key_mapped_value_a100);
  m.def(
      "shadowkv_fused_key_mapped_value_sm80_a100_v4(Tensor u, Tensor sv, Tensor! gathered_u, Tensor cosine, "
      "Tensor sine, Tensor component_kinds, Tensor source_slots, Tensor destination_slots, Tensor miss_ordinals, "
      "Tensor selected_chunk_ids, Tensor selected_lengths, Tensor plan_slots, Tensor! planner_error_codes, Tensor "
      "temporal_key_values, Tensor value_miss_chunk_ids, Tensor value_miss_lengths, Tensor descriptor_generation, "
      "Tensor descriptor_validity, int mapped_host_pointer, int mapped_host_bytes, int prompt_chunk_capacity, int "
      "prompt_tokens, int expected_generation, int plan_capacity, int reconstruction_stream, Tensor! "
      "destination_key_values) -> ()");
  m.impl(
      "shadowkv_fused_key_mapped_value_sm80_a100_v4",
      torch::kCUDA,
      &shadowkv_fused_key_mapped_value_staged_a100);
}
