"""Readable Torch oracles for every ShadowKV operation contract."""

from __future__ import annotations

from typing import Any

from sglang.kernels.ops.shadowkv.contracts import (
    ShadowKVReusePlan,
    validate_packed_gqa,
    validate_plan_reuse,
    validate_reconstruct,
    validate_reconstruct_rope,
)


def reconstruct(u: Any, sv: Any, positions: Any, out: Any | None = None) -> Any:
    """Gather low-rank factors and reconstruct positive-width keys."""
    torch = __import__("torch")
    device, expected_shape = validate_reconstruct(u, sv, positions, out)
    result = torch.einsum("hnr,hrd->hnd", u[positions], sv)
    if out is None:
        return result.reshape(expected_shape).to(device=device)
    out.copy_(result)
    return out


def reconstruct_rope(
    u: Any,
    sv: Any,
    positions: Any,
    inverse_frequencies: Any,
    out: Any | None = None,
) -> Any:
    """Reconstruct 64-element keys and apply NeoX Llama RoPE."""
    torch = __import__("torch")
    device, expected_shape = validate_reconstruct_rope(
        u, sv, positions, inverse_frequencies, out
    )
    reconstructed = torch.einsum("hnr,hrd->hnd", u[positions], sv).float()
    angles = positions.float().unsqueeze(-1) * inverse_frequencies
    cosine = torch.cat((angles.cos(), angles.cos()), dim=-1)
    sine = torch.cat((angles.sin(), angles.sin()), dim=-1)
    rotated_half = torch.cat(
        (-reconstructed[..., 32:], reconstructed[..., :32]), dim=-1
    )
    result = (reconstructed * cosine + rotated_half * sine).to(torch.bfloat16)
    if out is None:
        return result.reshape(expected_shape).to(device=device)
    out.copy_(result)
    return out


def packed_gqa(
    query: Any,
    keys: Any,
    values: Any,
    lengths: Any,
    *,
    weights: Any | None = None,
    out: Any | None = None,
    validate_lengths: bool = True,
) -> Any:
    """Compute readable ragged grouped-query attention in FP32."""
    torch = __import__("torch")
    device, weights_shape, out_shape = validate_packed_gqa(
        query,
        keys,
        values,
        lengths,
        weights=weights,
        out=out,
        validate_lengths=validate_lengths,
    )
    if weights is None:
        weights = torch.empty(weights_shape, dtype=torch.float32, device=device)
    if out is None:
        out = torch.empty(out_shape, dtype=torch.bfloat16, device=device)
    groups = query.shape[1] // keys.shape[1]
    head_dim = query.shape[-1]
    for row, length in enumerate(lengths.cpu().tolist()):
        weights[row].zero_()
        if length == 0:
            out[row].zero_()
            continue
        grouped = query[row].float().reshape(keys.shape[1], groups, head_dim)
        scores = torch.einsum("hgd,hkd->hgk", grouped, keys[row, :, :length].float())
        active_weights = torch.softmax(scores * (head_dim**-0.5), dim=-1)
        weights[row, :, :length].copy_(active_weights.reshape(query.shape[1], length))
        result = torch.einsum(
            "hgk,hkd->hgd", active_weights, values[row, :, :length].float()
        )
        out[row].copy_(result.reshape(query.shape[1], head_dim))
    return out


def plan_reuse(
    previous_chunks: Any,
    previous_lengths: Any,
    current_chunks: Any,
    current_lengths: Any,
    exact_chunks: Any,
    exact_lengths: Any,
    cached_generations: Any,
    current_generations: Any,
    *,
    max_reuse_chunks: int,
    chunk_size: int,
    validate: bool = True,
) -> ShadowKVReusePlan:
    """Produce the exact stable generation-aware reuse plan."""
    torch = __import__("torch")
    device, rows, current_width, exact_width = validate_plan_reuse(
        previous_chunks,
        previous_lengths,
        current_chunks,
        current_lengths,
        exact_chunks,
        exact_lengths,
        cached_generations,
        current_generations,
        max_reuse_chunks=max_reuse_chunks,
        chunk_size=chunk_size,
    )
    previous_width = previous_chunks.shape[1]
    previous_index = torch.arange(previous_width, device=device)
    current_index = torch.arange(current_width, device=device)
    exact_index = torch.arange(exact_width, device=device)
    previous_active = previous_index[None, :] < previous_lengths[:, None]
    current_active = current_index[None, :] < current_lengths[:, None]
    exact_active = exact_index[None, :] < exact_lengths[:, None]
    invalid_chunk = (
        ((previous_chunks < 0) & previous_active).any()
        | ((current_chunks < 0) & current_active).any()
        | ((exact_chunks < 0) & exact_active).any()
    )
    if validate and bool(invalid_chunk.item()):
        raise ValueError("reuse planner active chunks must be nonnegative")

    exact_prior = exact_index[None, :] < exact_index[:, None]
    exact_duplicate = (
        (exact_chunks[:, :, None] == exact_chunks[:, None, :])
        & exact_prior[None, :, :]
        & exact_active[:, None, :]
    ).any(dim=-1)
    exact_unique = exact_active & ~exact_duplicate
    exact_ranks = exact_unique.to(torch.int64).cumsum(dim=1) - 1
    deduplicated_storage = torch.full(
        (rows, exact_width + 1), -1, dtype=torch.int64, device=device
    )
    deduplicated_storage.scatter_(
        1,
        torch.where(exact_unique, exact_ranks, exact_width),
        torch.where(exact_unique, exact_chunks, -1),
    )
    deduplicated_exact = deduplicated_storage[:, :exact_width]

    current_prior = current_index[None, :] < current_index[:, None]
    current_duplicate = (
        (current_chunks[:, :, None] == current_chunks[:, None, :])
        & current_prior[None, :, :]
        & current_active[:, None, :]
    ).any(dim=-1)
    in_exact = (
        (current_chunks[:, :, None] == exact_chunks[:, None, :])
        & exact_active[:, None, :]
    ).any(dim=-1)
    reusable_previous = previous_active & (previous_index[None, :] < max_reuse_chunks)
    generation_matches = cached_generations == current_generations
    in_previous = (
        (current_chunks[:, :, None] == previous_chunks[:, None, :])
        & reusable_previous[:, None, :]
    ).any(dim=-1) & generation_matches[:, None]
    eligible = current_active & ~current_duplicate & ~in_exact
    hit = eligible & in_previous
    miss = eligible & ~in_previous
    kinds = torch.where(hit, 1, torch.where(miss, 2, 0)).to(torch.int64)
    kinds = torch.where(current_active, kinds, -1)
    miss_ranks = miss.to(torch.int64).cumsum(dim=1) - 1
    transfer_offsets = torch.where(miss, miss_ranks * chunk_size, -1)
    chunk_ids = torch.where(current_active, current_chunks, -1)
    plan = torch.stack((kinds, chunk_ids, transfer_offsets), dim=-1)
    counts = torch.stack(
        (hit.sum(dim=1), miss.sum(dim=1), exact_unique.sum(dim=1)), dim=1
    ).to(torch.int32)
    return ShadowKVReusePlan(plan, deduplicated_exact, counts)
