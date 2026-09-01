/* Copyright 2026 SGLang Team. All Rights Reserved. */

#pragma once

#include <ATen/Tensor.h>

#include <cstdint>

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
    at::Tensor& destination_key_values);

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
    at::Tensor& destination_key_values);

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
    at::Tensor& destination_key_values);

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
    at::Tensor& destination_key_values);

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
    at::Tensor& destination_key_values);
