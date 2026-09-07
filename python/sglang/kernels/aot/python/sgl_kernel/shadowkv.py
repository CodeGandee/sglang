"""Wrappers for the four optional ShadowKV CUDA operations."""

from dataclasses import dataclass

import torch

_OPERATOR_NAMES = (
    "shadowkv_reconstruct_generic_aot_v1",
    "shadowkv_reconstruct_rope_generic_aot_v1",
    "shadowkv_plan_reuse_generic_aot_v1",
    "shadowkv_packed_gqa_generic_aot_v1",
)


@dataclass(frozen=True)
class ShadowKVReusePlan:
    plan: torch.Tensor
    deduplicated_exact_chunks: torch.Tensor
    counts: torch.Tensor

    @property
    def kinds(self) -> torch.Tensor:
        return self.plan[..., 0]

    @property
    def chunk_ids(self) -> torch.Tensor:
        return self.plan[..., 1]

    @property
    def transfer_offsets(self) -> torch.Tensor:
        return self.plan[..., 2]


def shadowkv_kernels_available() -> bool:
    return all(hasattr(torch.ops.sgl_kernel, name) for name in _OPERATOR_NAMES)


def _require_supported_device(device: torch.device) -> None:
    if not shadowkv_kernels_available():
        raise RuntimeError("the installed wheel has no ShadowKV kernels")
    if not torch.cuda.is_available():
        raise RuntimeError("ShadowKV kernels require a visible NVIDIA GPU")
    capability = torch.cuda.get_device_capability(device)
    if capability not in {(8, 0), (10, 0)}:
        raise RuntimeError(
            "ShadowKV kernels require compute capability 8.0 or 10.0; "
            f"found {capability}"
        )


def _require_tensor(
    name: str, tensor: torch.Tensor, dtype: torch.dtype, dimensions: int
) -> None:
    if tensor.dtype != dtype or tensor.ndim != dimensions:
        raise ValueError(f"{name} must be a {dimensions}D {dtype} tensor")
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError(f"{name} must be a contiguous CUDA tensor")


def shadowkv_reconstruct(
    u: torch.Tensor,
    sv: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    _require_tensor("u", u, torch.bfloat16, 2)
    _require_tensor("sv", sv, torch.bfloat16, 3)
    _require_tensor("positions", positions, torch.int64, 2)
    if u.shape[1] not in (64, 128, 160, 256):
        raise ValueError("rank must be one of 64, 128, 160, or 256")
    if sv.shape[1:] != (u.shape[1], 128):
        raise ValueError("sv must have shape [kv_heads, rank, 128]")
    if positions.shape[0] != sv.shape[0]:
        raise ValueError("positions and sv must have the same kv_heads")
    if any(value.device != u.device for value in (sv, positions)):
        raise ValueError("reconstruction tensors must share one CUDA device")
    _require_supported_device(u.device)
    expected = (sv.shape[0], positions.shape[1], 128)
    out = (
        torch.empty(expected, dtype=torch.bfloat16, device=u.device)
        if out is None
        else out
    )
    if out.shape != expected or out.dtype != torch.bfloat16 or out.device != u.device:
        raise ValueError(f"out must have shape {expected} on {u.device}")
    torch.ops.sgl_kernel.shadowkv_reconstruct_generic_aot_v1.default(
        u, sv, positions, out
    )
    return out


def shadowkv_reconstruct_rope(
    u: torch.Tensor,
    sv: torch.Tensor,
    positions: torch.Tensor,
    inverse_frequencies: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    _require_tensor("u", u, torch.bfloat16, 2)
    _require_tensor("sv", sv, torch.bfloat16, 3)
    _require_tensor("positions", positions, torch.int64, 2)
    _require_tensor("inverse_frequencies", inverse_frequencies, torch.float32, 1)
    if u.shape[1] != 160 or sv.shape[1:] != (160, 64):
        raise ValueError("RoPE reconstruction requires rank 160 and head dimension 64")
    if positions.shape[0] != sv.shape[0] or inverse_frequencies.shape != (32,):
        raise ValueError("RoPE reconstruction shapes are incompatible")
    if any(value.device != u.device for value in (sv, positions, inverse_frequencies)):
        raise ValueError("reconstruction tensors must share one CUDA device")
    _require_supported_device(u.device)
    expected = (sv.shape[0], positions.shape[1], 64)
    out = (
        torch.empty(expected, dtype=torch.bfloat16, device=u.device)
        if out is None
        else out
    )
    if out.shape != expected or out.dtype != torch.bfloat16 or out.device != u.device:
        raise ValueError(f"out must have shape {expected} on {u.device}")
    torch.ops.sgl_kernel.shadowkv_reconstruct_rope_generic_aot_v1.default(
        u, sv, positions, inverse_frequencies, out
    )
    return out


def shadowkv_packed_gqa(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    lengths: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    validate_lengths: bool = True,
) -> torch.Tensor:
    _require_tensor("query", query, torch.bfloat16, 3)
    _require_tensor("keys", keys, torch.bfloat16, 4)
    _require_tensor("values", values, torch.bfloat16, 4)
    _require_tensor("lengths", lengths, torch.int32, 1)
    if query.shape[-1] not in (64, 128) or keys.shape != values.shape:
        raise ValueError("packed GQA tensor shapes are incompatible")
    if query.shape[0] != keys.shape[0] or query.shape[1] % keys.shape[1]:
        raise ValueError("packed GQA batch or head grouping is incompatible")
    if lengths.shape != (query.shape[0],):
        raise ValueError("lengths must have shape [batch]")
    if any(value.device != query.device for value in (keys, values, lengths)):
        raise ValueError("packed GQA tensors must share one CUDA device")
    if (
        validate_lengths
        and lengths.numel()
        and (int(lengths.min().item()) < 0 or int(lengths.max().item()) > keys.shape[2])
    ):
        raise ValueError("lengths exceed the packed KV token capacity")
    _require_supported_device(query.device)
    weight_shape = (query.shape[0], query.shape[1], keys.shape[2])
    weights = (
        torch.empty(weight_shape, dtype=torch.float32, device=query.device)
        if weights is None
        else weights
    )
    out = torch.empty_like(query) if out is None else out
    if weights.shape != weight_shape or out.shape != query.shape:
        raise ValueError("caller-owned packed GQA outputs have incompatible shapes")
    torch.ops.sgl_kernel.shadowkv_packed_gqa_generic_aot_v1.default(
        query, keys, values, lengths, weights, out
    )
    return out


def shadowkv_plan_reuse(
    previous_chunks: torch.Tensor,
    previous_lengths: torch.Tensor,
    current_chunks: torch.Tensor,
    current_lengths: torch.Tensor,
    exact_chunks: torch.Tensor,
    exact_lengths: torch.Tensor,
    cached_generations: torch.Tensor,
    current_generations: torch.Tensor,
    *,
    max_reuse_chunks: int,
    chunk_size: int,
    validate: bool = True,
) -> ShadowKVReusePlan:
    chunks = (previous_chunks, current_chunks, exact_chunks)
    lengths = (previous_lengths, current_lengths, exact_lengths)
    generations = (cached_generations, current_generations)
    for name, tensor in zip(
        ("previous_chunks", "current_chunks", "exact_chunks"), chunks, strict=True
    ):
        _require_tensor(name, tensor, torch.int64, 2)
    for name, tensor in zip(
        ("previous_lengths", "current_lengths", "exact_lengths"), lengths, strict=True
    ):
        _require_tensor(name, tensor, torch.int32, 1)
    for name, tensor in zip(
        ("cached_generations", "current_generations"), generations, strict=True
    ):
        _require_tensor(name, tensor, torch.int64, 1)
    rows, width = current_chunks.shape
    if any(tensor.shape[0] != rows for tensor in chunks):
        raise ValueError("reuse planner rows differ")
    if any(tensor.shape != (rows,) for tensor in (*lengths, *generations)):
        raise ValueError("reuse planner metadata must have shape [rows]")
    if any(
        tensor.device != current_chunks.device
        for tensor in (*chunks, *lengths, *generations)
    ):
        raise ValueError("reuse planner tensors must share one CUDA device")
    if not 0 <= max_reuse_chunks <= previous_chunks.shape[1] or chunk_size < 1:
        raise ValueError("reuse planner bounds are invalid")
    _require_supported_device(current_chunks.device)
    plan = torch.full(
        (rows, width, 3), -1, dtype=torch.int64, device=current_chunks.device
    )
    deduplicated = torch.full_like(exact_chunks, -1)
    counts = torch.zeros((rows, 3), dtype=torch.int32, device=current_chunks.device)
    errors = torch.zeros((rows,), dtype=torch.int32, device=current_chunks.device)
    torch.ops.sgl_kernel.shadowkv_plan_reuse_generic_aot_v1.default(
        previous_chunks,
        previous_lengths,
        current_chunks,
        current_lengths,
        exact_chunks,
        exact_lengths,
        cached_generations,
        current_generations,
        max_reuse_chunks,
        chunk_size,
        plan,
        deduplicated,
        counts,
        errors,
    )
    if validate and bool(errors.any().item()):
        raise ValueError(f"invalid ShadowKV reuse plan input: {errors.cpu().tolist()}")
    return ShadowKVReusePlan(plan, deduplicated, counts)
