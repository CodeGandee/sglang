"""Shared validation and result types for ShadowKV operation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ShadowKVReusePlan:
    """Stable row-packed reuse plan shared by every provider."""

    plan: Any
    deduplicated_exact_chunks: Any
    counts: Any

    @property
    def kinds(self) -> Any:
        return self.plan[..., 0]

    @property
    def chunk_ids(self) -> Any:
        return self.plan[..., 1]

    @property
    def transfer_offsets(self) -> Any:
        return self.plan[..., 2]


def _torch():
    import torch

    return torch


def require_tensor(
    name: str,
    tensor: Any,
    *,
    dtype: Any,
    dimensions: int,
    contiguous: bool = True,
) -> None:
    torch = _torch()
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must use {dtype}")
    if tensor.ndim != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions")
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def validate_reconstruct(
    u: Any, sv: Any, positions: Any, out: Any | None
) -> tuple[Any, tuple[int, int, int]]:
    torch = _torch()
    require_tensor("u", u, dtype=torch.bfloat16, dimensions=2)
    require_tensor("sv", sv, dtype=torch.bfloat16, dimensions=3)
    require_tensor("positions", positions, dtype=torch.int64, dimensions=2)
    approved_ranks = (64, 128, 160, 256)
    if u.shape[1] not in approved_ranks:
        raise ValueError("rank must be one of 64, 128, 160, or 256")
    if sv.shape[1] != u.shape[1] or sv.shape[2] < 1:
        raise ValueError("sv must have shape [kv_heads, rank, positive head_dim]")
    if positions.shape[0] != sv.shape[0]:
        raise ValueError("positions and sv must have the same kv_heads")
    if u.shape[0] < 1:
        raise ValueError("u must contain at least one token")
    device = u.device
    if any(tensor.device != device for tensor in (sv, positions)):
        raise ValueError("all reconstruction tensors must share one device")
    if positions.numel() and (
        int(positions.min().item()) < 0 or int(positions.max().item()) >= u.shape[0]
    ):
        raise ValueError("positions exceed the U token dimension")
    expected_shape = (sv.shape[0], positions.shape[1], sv.shape[2])
    if out is not None:
        require_tensor(
            "out",
            out,
            dtype=torch.bfloat16,
            dimensions=3,
            contiguous=False,
        )
        if out.stride(-1) != 1:
            raise ValueError("out must be contiguous in its head dimension")
        if out.shape != expected_shape:
            raise ValueError(f"out must have shape {expected_shape}")
        if out.device != device:
            raise ValueError("out must share the reconstruction device")
    return device, expected_shape


def validate_reconstruct_rope(
    u: Any,
    sv: Any,
    positions: Any,
    inverse_frequencies: Any,
    out: Any | None,
) -> tuple[Any, tuple[int, int, int]]:
    torch = _torch()
    require_tensor("u", u, dtype=torch.bfloat16, dimensions=2)
    require_tensor("sv", sv, dtype=torch.bfloat16, dimensions=3)
    require_tensor("positions", positions, dtype=torch.int64, dimensions=2)
    require_tensor(
        "inverse_frequencies",
        inverse_frequencies,
        dtype=torch.float32,
        dimensions=1,
    )
    if u.shape[1] != 160:
        raise ValueError("u must have shape [tokens, 160]")
    if sv.shape[1:] != (160, 64):
        raise ValueError("sv must have shape [kv_heads, 160, 64]")
    if positions.shape[0] != sv.shape[0]:
        raise ValueError("positions and sv must have the same kv_heads")
    if inverse_frequencies.shape != (32,):
        raise ValueError("inverse_frequencies must have shape [32]")
    if u.shape[0] < 1:
        raise ValueError("u must contain at least one token")
    device = u.device
    if any(tensor.device != device for tensor in (sv, positions, inverse_frequencies)):
        raise ValueError("all reconstruction tensors must share one device")
    if positions.numel() and (
        int(positions.min().item()) < 0 or int(positions.max().item()) >= u.shape[0]
    ):
        raise ValueError("positions exceed the U token dimension")
    expected_shape = (sv.shape[0], positions.shape[1], 64)
    if out is not None:
        require_tensor(
            "out",
            out,
            dtype=torch.bfloat16,
            dimensions=3,
            contiguous=False,
        )
        if out.stride(-1) != 1:
            raise ValueError("out must be contiguous in its head dimension")
        if out.shape != expected_shape:
            raise ValueError(f"out must have shape {expected_shape}")
        if out.device != device:
            raise ValueError("out must share the reconstruction device")
    return device, expected_shape


def validate_packed_gqa(
    query: Any,
    keys: Any,
    values: Any,
    lengths: Any,
    *,
    weights: Any | None,
    out: Any | None,
    validate_lengths: bool,
) -> tuple[Any, tuple[int, int, int], tuple[int, int, int]]:
    torch = _torch()
    require_tensor("query", query, dtype=torch.bfloat16, dimensions=3)
    require_tensor("keys", keys, dtype=torch.bfloat16, dimensions=4)
    require_tensor("values", values, dtype=torch.bfloat16, dimensions=4)
    require_tensor("lengths", lengths, dtype=torch.int32, dimensions=1)
    if query.shape[-1] not in (64, 128):
        raise ValueError("query head dimension must be 64 or 128")
    if keys.shape[-1] != query.shape[-1]:
        raise ValueError("keys must match the query head dimension")
    if values.shape != keys.shape:
        raise ValueError("values must match the packed key shape")
    if query.shape[0] != keys.shape[0]:
        raise ValueError("query and packed KV batch dimensions differ")
    if (
        query.shape[0] < 1
        or query.shape[1] < 1
        or keys.shape[1] < 1
        or keys.shape[2] < 1
    ):
        raise ValueError("packed GQA dimensions must be positive")
    if query.shape[1] % keys.shape[1]:
        raise ValueError("query heads must be divisible by KV heads")
    if lengths.shape != (query.shape[0],):
        raise ValueError("lengths must have shape [batch]")
    device = query.device
    if any(tensor.device != device for tensor in (keys, values, lengths)):
        raise ValueError("all packed GQA inputs must share one device")
    if (
        validate_lengths
        and lengths.numel()
        and (int(lengths.min().item()) < 0 or int(lengths.max().item()) > keys.shape[2])
    ):
        raise ValueError("lengths exceed the packed KV token capacity")
    expected_weights = (query.shape[0], query.shape[1], keys.shape[2])
    if weights is not None:
        require_tensor("weights", weights, dtype=torch.float32, dimensions=3)
        if weights.shape != expected_weights or weights.device != device:
            raise ValueError(f"weights must have shape {expected_weights} on {device}")
    expected_out = tuple(query.shape)
    if out is not None:
        require_tensor("out", out, dtype=torch.bfloat16, dimensions=3)
        if out.shape != query.shape or out.device != device:
            raise ValueError(f"out must have shape {query.shape} on {device}")
    return device, expected_weights, expected_out


def validate_plan_reuse(
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
) -> tuple[Any, int, int, int]:
    torch = _torch()
    chunks = (previous_chunks, current_chunks, exact_chunks)
    lengths = (previous_lengths, current_lengths, exact_lengths)
    generations = (cached_generations, current_generations)
    for name, tensor in zip(
        ("previous_chunks", "current_chunks", "exact_chunks"),
        chunks,
        strict=True,
    ):
        require_tensor(name, tensor, dtype=torch.int64, dimensions=2)
    for name, tensor in zip(
        ("previous_lengths", "current_lengths", "exact_lengths"),
        lengths,
        strict=True,
    ):
        require_tensor(name, tensor, dtype=torch.int32, dimensions=1)
    for name, tensor in zip(
        ("cached_generations", "current_generations"),
        generations,
        strict=True,
    ):
        require_tensor(name, tensor, dtype=torch.int64, dimensions=1)
    rows, current_width = current_chunks.shape
    if any(tensor.shape[0] != rows for tensor in chunks):
        raise ValueError("all planner chunk tensors need equal row counts")
    if any(tensor.shape != (rows,) for tensor in (*lengths, *generations)):
        raise ValueError("all planner metadata tensors must have shape [rows]")
    device = current_chunks.device
    if any(tensor.device != device for tensor in (*chunks, *lengths, *generations)):
        raise ValueError("all planner tensors must share one device")
    if max_reuse_chunks < 0 or max_reuse_chunks > previous_chunks.shape[1]:
        raise ValueError("max_reuse_chunks exceeds the previous width")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if previous_chunks.shape[1] > 256 or current_width > 256:
        raise ValueError("planner chunk widths must not exceed 256")
    if exact_chunks.shape[1] > 64:
        raise ValueError("planner exact width must not exceed 64")
    length_rows = torch.stack(lengths, dim=1).to(dtype=torch.int64)
    widths = torch.tensor(
        (previous_chunks.shape[1], current_width, exact_chunks.shape[1]),
        dtype=torch.int64,
        device=device,
    )
    if bool(((length_rows < 0) | (length_rows > widths)).any().item()):
        raise ValueError("reuse planner length exceeds its row width")
    return device, rows, current_width, exact_chunks.shape[1]
