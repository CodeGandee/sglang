"""Optional SM80 and SM100a ShadowKV AOT kernel wrappers."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ShadowKVReusePlan:
    """Stable row-packed reuse plan produced by the AOT planner."""

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
    """Return whether this wheel contains the optional ShadowKV operators."""

    return (
        hasattr(torch.ops.sgl_kernel, "shadowkv_reconstruct")
        and hasattr(torch.ops.sgl_kernel, "shadowkv_reconstruct_rope")
        and hasattr(torch.ops.sgl_kernel, "shadowkv_plan_reuse")
        and hasattr(torch.ops.sgl_kernel, "shadowkv_packed_gqa")
    )


def _launch_shadowkv_packed_gqa(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    lengths: torch.Tensor,
    weights: torch.Tensor,
    out: torch.Tensor,
) -> None:
    torch.ops.sgl_kernel.shadowkv_packed_gqa.default(
        query,
        keys,
        values,
        lengths,
        weights,
        out,
    )


def _launch_shadowkv_reconstruct_rope(
    u: torch.Tensor,
    sv: torch.Tensor,
    positions: torch.Tensor,
    inverse_frequencies: torch.Tensor,
    out: torch.Tensor,
) -> None:
    torch.ops.sgl_kernel.shadowkv_reconstruct_rope.default(
        u, sv, positions, inverse_frequencies, out
    )


def _launch_shadowkv_reconstruct(
    u: torch.Tensor,
    sv: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
) -> None:
    torch.ops.sgl_kernel.shadowkv_reconstruct.default(u, sv, positions, out)


def _launch_shadowkv_plan_reuse(
    previous_chunks: torch.Tensor,
    previous_lengths: torch.Tensor,
    current_chunks: torch.Tensor,
    current_lengths: torch.Tensor,
    exact_chunks: torch.Tensor,
    exact_lengths: torch.Tensor,
    cached_generations: torch.Tensor,
    current_generations: torch.Tensor,
    max_reuse_chunks: int,
    chunk_size: int,
    plan: torch.Tensor,
    deduplicated_exact: torch.Tensor,
    counts: torch.Tensor,
    error_codes: torch.Tensor,
) -> None:
    torch.ops.sgl_kernel.shadowkv_plan_reuse.default(
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
        deduplicated_exact,
        counts,
        error_codes,
    )


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
    """Run caller-buffered ragged GQA over fixed-stride packed storage."""

    _require_tensor("query", query, dtype=torch.bfloat16, dimensions=3)
    _require_tensor("keys", keys, dtype=torch.bfloat16, dimensions=4)
    _require_tensor("values", values, dtype=torch.bfloat16, dimensions=4)
    _require_tensor("lengths", lengths, dtype=torch.int32, dimensions=1)
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
        raise ValueError("all packed GQA inputs must share one CUDA device")
    if (
        validate_lengths
        and lengths.numel()
        and (int(lengths.min().item()) < 0 or int(lengths.max().item()) > keys.shape[2])
    ):
        raise ValueError("lengths exceed the packed KV token capacity")
    _require_supported_device(device)
    expected_weights = (query.shape[0], query.shape[1], keys.shape[2])
    if weights is None:
        weights = torch.empty(expected_weights, dtype=torch.float32, device=device)
    else:
        _require_tensor("weights", weights, dtype=torch.float32, dimensions=3)
        if weights.shape != expected_weights or weights.device != device:
            raise ValueError(f"weights must have shape {expected_weights} on {device}")
    if out is None:
        out = torch.empty_like(query)
    else:
        _require_tensor("out", out, dtype=torch.bfloat16, dimensions=3)
        if out.shape != query.shape or out.device != device:
            raise ValueError(f"out must have shape {query.shape} on {device}")
    _launch_shadowkv_packed_gqa(
        query,
        keys,
        values,
        lengths,
        weights,
        out,
    )
    return out


def _require_supported_device(device: torch.device) -> None:
    if not shadowkv_kernels_available():
        raise RuntimeError(
            "the installed sglang-kernel wheel was built without optional ShadowKV kernels"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("ShadowKV AOT kernels require a visible NVIDIA GPU")
    capability = torch.cuda.get_device_capability(device)
    if capability not in {(8, 0), (10, 0)}:
        raise RuntimeError(
            "ShadowKV AOT kernels require compute capability 8.0 or 10.0; "
            f"found {capability}"
        )


def _require_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    dimensions: int,
) -> None:
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must use {dtype}")
    if tensor.ndim != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def shadowkv_reconstruct_rope(
    u: torch.Tensor,
    sv: torch.Tensor,
    positions: torch.Tensor,
    inverse_frequencies: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather U, reconstruct per-head keys, and apply NeoX Llama RoPE."""

    _require_tensor("u", u, dtype=torch.bfloat16, dimensions=2)
    _require_tensor("sv", sv, dtype=torch.bfloat16, dimensions=3)
    _require_tensor("positions", positions, dtype=torch.int64, dimensions=2)
    _require_tensor(
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
        raise ValueError("all reconstruction tensors must share one CUDA device")
    if positions.numel() and (
        int(positions.min().item()) < 0 or int(positions.max().item()) >= u.shape[0]
    ):
        raise ValueError("positions exceed the U token dimension")
    _require_supported_device(device)
    expected_shape = (sv.shape[0], positions.shape[1], 64)
    if out is None:
        out = torch.empty(expected_shape, dtype=torch.bfloat16, device=u.device)
    else:
        if out.dtype != torch.bfloat16 or out.ndim != 3 or not out.is_cuda:
            raise ValueError("out must be a 3D CUDA bfloat16 tensor")
        if out.stride(-1) != 1:
            raise ValueError("out must be contiguous in its head dimension")
        if out.shape != expected_shape:
            raise ValueError(f"out must have shape {expected_shape}")
        if out.device != device:
            raise ValueError("out must share the reconstruction CUDA device")
    _launch_shadowkv_reconstruct_rope(u, sv, positions, inverse_frequencies, out)
    return out


def shadowkv_reconstruct(
    u: torch.Tensor,
    sv: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather U and reconstruct 128-element pre-RoPE keys."""

    _require_tensor("u", u, dtype=torch.bfloat16, dimensions=2)
    _require_tensor("sv", sv, dtype=torch.bfloat16, dimensions=3)
    _require_tensor("positions", positions, dtype=torch.int64, dimensions=2)
    approved_ranks = (64, 128, 160, 256)
    if u.shape[1] not in approved_ranks:
        raise ValueError("rank must be one of 64, 128, 160, or 256")
    if sv.shape[1:] != (u.shape[1], 128):
        raise ValueError("sv must have shape [kv_heads, rank, 128]")
    if positions.shape[0] != sv.shape[0]:
        raise ValueError("positions and sv must have the same kv_heads")
    if u.shape[0] < 1:
        raise ValueError("u must contain at least one token")
    device = u.device
    if any(tensor.device != device for tensor in (sv, positions)):
        raise ValueError("all reconstruction tensors must share one CUDA device")
    if positions.numel() and (
        int(positions.min().item()) < 0 or int(positions.max().item()) >= u.shape[0]
    ):
        raise ValueError("positions exceed the U token dimension")
    _require_supported_device(device)
    expected_shape = (sv.shape[0], positions.shape[1], 128)
    if out is None:
        out = torch.empty(expected_shape, dtype=torch.bfloat16, device=device)
    else:
        if out.dtype != torch.bfloat16 or out.ndim != 3 or not out.is_cuda:
            raise ValueError("out must be a 3D CUDA bfloat16 tensor")
        if out.stride(-1) != 1:
            raise ValueError("out must be contiguous in its head dimension")
        if out.shape != expected_shape:
            raise ValueError(f"out must have shape {expected_shape}")
        if out.device != device:
            raise ValueError("out must share the reconstruction CUDA device")
    _launch_shadowkv_reconstruct(u, sv, positions, out)
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
    """Plan stable hit, miss, exact-deduplication, and transfer-offset regions."""

    chunks = (previous_chunks, current_chunks, exact_chunks)
    lengths = (previous_lengths, current_lengths, exact_lengths)
    generations = (cached_generations, current_generations)
    for name, tensor in zip(
        ("previous_chunks", "current_chunks", "exact_chunks"),
        chunks,
        strict=True,
    ):
        _require_tensor(name, tensor, dtype=torch.int64, dimensions=2)
    for name, tensor in zip(
        ("previous_lengths", "current_lengths", "exact_lengths"),
        lengths,
        strict=True,
    ):
        _require_tensor(name, tensor, dtype=torch.int32, dimensions=1)
    for name, tensor in zip(
        ("cached_generations", "current_generations"),
        generations,
        strict=True,
    ):
        _require_tensor(name, tensor, dtype=torch.int64, dimensions=1)
    rows, current_width = current_chunks.shape
    if any(tensor.shape[0] != rows for tensor in chunks):
        raise ValueError("all planner chunk tensors need equal row counts")
    if any(tensor.shape != (rows,) for tensor in (*lengths, *generations)):
        raise ValueError("all planner metadata tensors must have shape [rows]")
    device = current_chunks.device
    if any(tensor.device != device for tensor in (*chunks, *lengths, *generations)):
        raise ValueError("all planner tensors must share one CUDA device")
    if max_reuse_chunks < 0 or max_reuse_chunks > previous_chunks.shape[1]:
        raise ValueError("max_reuse_chunks exceeds the previous width")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if previous_chunks.shape[1] > 256 or current_width > 256:
        raise ValueError("planner chunk widths must not exceed 256")
    if exact_chunks.shape[1] > 64:
        raise ValueError("planner exact width must not exceed 64")
    _require_supported_device(device)
    plan = torch.full(
        (rows, current_width, 3),
        -1,
        dtype=torch.int64,
        device=current_chunks.device,
    )
    deduplicated_exact = torch.full_like(exact_chunks, -1)
    counts = torch.zeros((rows, 3), dtype=torch.int32, device=current_chunks.device)
    error_codes = torch.zeros((rows,), dtype=torch.int32, device=current_chunks.device)
    _launch_shadowkv_plan_reuse(
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
        deduplicated_exact,
        counts,
        error_codes,
    )
    if validate:
        errors = error_codes.cpu().tolist()
        if any(errors):
            details = ", ".join(
                f"row {row}: code {code}" for row, code in enumerate(errors) if code
            )
            raise ValueError(f"invalid ShadowKV reuse plan input ({details})")
    return ShadowKVReusePlan(plan, deduplicated_exact, counts)
