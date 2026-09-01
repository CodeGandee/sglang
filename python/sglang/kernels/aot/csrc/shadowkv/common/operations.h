/* Copyright 2026 SGLang Team. All Rights Reserved. */

#pragma once

#include <ATen/Tensor.h>

#include <cstdint>

void shadowkv_reconstruct(
    const at::Tensor& u,
    const at::Tensor& sv,
    const at::Tensor& positions,
    at::Tensor& output);
void shadowkv_reconstruct_rope(
    const at::Tensor& u,
    const at::Tensor& sv,
    const at::Tensor& positions,
    const at::Tensor& inverse_frequencies,
    at::Tensor& output);
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
    at::Tensor& error_codes);
void shadowkv_packed_gqa(
    const at::Tensor& query,
    const at::Tensor& keys,
    const at::Tensor& values,
    const at::Tensor& lengths,
    at::Tensor& weights,
    at::Tensor& output);
