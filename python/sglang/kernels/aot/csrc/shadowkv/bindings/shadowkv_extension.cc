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
  m.def(
      "shadowkv_plan_device_generic_aot_v1(Tensor selected_chunk_ids, Tensor selected_lengths, Tensor "
      "exact_chunk_ids, Tensor exact_lengths, Tensor temporal_chunk_ids, Tensor temporal_component_validity, "
      "Tensor temporal_publication_generations, Tensor temporal_request_generations, Tensor "
      "temporal_layout_generations, Tensor row_indices, Tensor row_generations, Tensor plan_slots, int "
      "plan_capacity, Tensor! component_kinds, Tensor! source_slots, Tensor! destination_slots, Tensor! "
      "miss_ordinals, Tensor! counts, Tensor! error_codes) -> ()");
  m.impl("shadowkv_plan_device_generic_aot_v1", torch::kCUDA, &shadowkv_plan_device);
  m.def(
      "shadowkv_plan_device_v2_generic_aot_v1(Tensor selected_chunk_ids, Tensor selected_lengths, Tensor "
      "exact_chunk_ids, Tensor exact_lengths, Tensor temporal_chunk_ids, Tensor temporal_component_validity, "
      "Tensor temporal_publication_generations, Tensor temporal_request_generations, Tensor "
      "temporal_layout_generations, Tensor row_indices, Tensor row_generations, Tensor plan_slots, int "
      "plan_capacity, Tensor! component_kinds, Tensor! source_slots, Tensor! destination_slots, Tensor! "
      "miss_ordinals, Tensor! counts, Tensor! error_codes, Tensor! value_miss_chunk_ids, Tensor! "
      "value_miss_lengths) -> ()");
  m.impl("shadowkv_plan_device_v2_generic_aot_v1", torch::kCUDA, &shadowkv_plan_device_v2);
  m.def(
      "shadowkv_publish_value_descriptor_generic_aot_v1(Tensor! descriptor_generation, Tensor! "
      "descriptor_validity, int generation) -> ()");
  m.impl(
      "shadowkv_publish_value_descriptor_generic_aot_v1", torch::kCUDA, &shadowkv_publish_value_descriptor);
  m.def("shadowkv_resolve_mapped_host_pointer_generic_aot_v1(Tensor host_values, int device_index) -> int");
  m.impl(
      "shadowkv_resolve_mapped_host_pointer_generic_aot_v1", torch::kCPU, &shadowkv_resolve_mapped_host_pointer);
  m.def(
      "shadowkv_place_device_generic_aot_v1(Tensor component_kinds, Tensor source_slots, Tensor "
      "destination_slots, Tensor plan_slots, Tensor! planner_error_codes, Tensor temporal_key_values, Tensor "
      "compatibility_key_values, int plan_capacity, Tensor! destination_key_values) -> ()");
  m.impl("shadowkv_place_device_generic_aot_v1", torch::kCUDA, &shadowkv_place_device);
  m.def(
      "shadowkv_place_device_miss_only_generic_aot_v1(Tensor component_kinds, Tensor source_slots, Tensor "
      "destination_slots, Tensor miss_ordinals, Tensor selected_chunk_ids, Tensor plan_slots, Tensor! "
      "planner_error_codes, Tensor temporal_key_values, Tensor reconstructed_keys, Tensor value_miss_chunk_ids, "
      "Tensor value_miss_lengths, Tensor descriptor_generation, Tensor descriptor_validity, int "
      "expected_generation, int plan_capacity, Tensor value_miss_key_values, Tensor! destination_key_values) "
      "-> ()");
  m.impl("shadowkv_place_device_miss_only_generic_aot_v1", torch::kCUDA, &shadowkv_place_device_miss_only);
  m.def(
      "shadowkv_place_device_mapped_host_generic_aot_v1(Tensor component_kinds, Tensor source_slots, Tensor "
      "destination_slots, Tensor miss_ordinals, Tensor selected_chunk_ids, Tensor plan_slots, Tensor! "
      "planner_error_codes, Tensor temporal_key_values, Tensor reconstructed_keys, Tensor value_miss_chunk_ids, "
      "Tensor value_miss_lengths, Tensor descriptor_generation, Tensor descriptor_validity, int "
      "mapped_host_pointer, int mapped_host_bytes, int prompt_chunk_capacity, int prompt_tokens, int "
      "expected_generation, int plan_capacity, Tensor! destination_key_values) -> ()");
  m.impl("shadowkv_place_device_mapped_host_generic_aot_v1", torch::kCUDA, &shadowkv_place_device_mapped_host);
  m.def(
      "shadowkv_publish_device_generic_aot_v1(Tensor selected_chunk_ids, Tensor selected_lengths, Tensor "
      "exact_chunk_ids, Tensor exact_lengths, Tensor row_indices, Tensor row_generations, Tensor "
      "planner_error_codes, Tensor destination_key_values, Tensor temporal_request_generations, Tensor "
      "temporal_layout_generations, Tensor! temporal_chunk_ids, Tensor! temporal_key_values, Tensor! "
      "temporal_publication_generations, Tensor! temporal_component_validity) -> ()");
  m.impl("shadowkv_publish_device_generic_aot_v1", torch::kCUDA, &shadowkv_publish_device);
}

REGISTER_EXTENSION(shadowkv_ops)
