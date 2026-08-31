"""Optional B200 ShadowKV AOT kernel wrappers."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ShadowKVReusePlan:
    """Stable row-packed reuse plan produced by the B200 planner."""

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


class ShadowKVDevicePlan:
    """Compact component-aware plan whose validation remains on the device."""

    __slots__ = (
        "component_kinds",
        "counts",
        "destination_slots",
        "error_codes",
        "miss_ordinals",
        "plan_slots",
        "row_generations",
        "row_indices",
        "selected_chunk_ids",
        "selected_lengths",
        "source_slots",
    )

    def __init__(
        self,
        *,
        row_indices: torch.Tensor,
        row_generations: torch.Tensor,
        selected_chunk_ids: torch.Tensor,
        selected_lengths: torch.Tensor,
        plan_slots: torch.Tensor,
        component_kinds: torch.Tensor,
        source_slots: torch.Tensor,
        destination_slots: torch.Tensor,
        miss_ordinals: torch.Tensor,
        counts: torch.Tensor,
        error_codes: torch.Tensor,
    ) -> None:
        self.row_indices = row_indices
        self.row_generations = row_generations
        self.selected_chunk_ids = selected_chunk_ids
        self.selected_lengths = selected_lengths
        self.plan_slots = plan_slots
        self.component_kinds = component_kinds
        self.source_slots = source_slots
        self.destination_slots = destination_slots
        self.miss_ordinals = miss_ordinals
        self.counts = counts
        self.error_codes = error_codes


@dataclass(frozen=True)
class ShadowKVDevicePlanOutputs:
    """Caller-owned output tensors for allocation-free device planning."""

    component_kinds: torch.Tensor
    source_slots: torch.Tensor
    destination_slots: torch.Tensor
    miss_ordinals: torch.Tensor
    counts: torch.Tensor
    error_codes: torch.Tensor


def shadowkv_kernels_available() -> bool:
    """Return whether this wheel contains the optional ShadowKV operators."""

    return (
        hasattr(torch.ops.sgl_kernel, "shadowkv_reconstruct")
        and hasattr(torch.ops.sgl_kernel, "shadowkv_reconstruct_rope")
        and hasattr(torch.ops.sgl_kernel, "shadowkv_place_device")
        and hasattr(torch.ops.sgl_kernel, "shadowkv_plan_device")
        and hasattr(torch.ops.sgl_kernel, "shadowkv_plan_reuse")
        and hasattr(torch.ops.sgl_kernel, "shadowkv_publish_device")
        and hasattr(torch.ops.sgl_kernel, "shadowkv_packed_gqa")
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
    """Run caller-buffered ragged GQA over fixed-stride packed B200 storage."""

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
    _require_b200(device)
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
    torch.ops.sgl_kernel.shadowkv_packed_gqa.default(
        query,
        keys,
        values,
        lengths,
        weights,
        out,
    )
    return out


def _require_b200(device: torch.device) -> None:
    if not shadowkv_kernels_available():
        raise RuntimeError(
            "the installed sglang-kernel wheel was built without optional ShadowKV kernels"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("ShadowKV AOT kernels require a visible NVIDIA B200")
    capability = torch.cuda.get_device_capability(device)
    if capability != (10, 0):
        raise RuntimeError(
            "ShadowKV AOT kernels require NVIDIA B200 compute capability 10.0; "
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
    _require_b200(device)
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
    torch.ops.sgl_kernel.shadowkv_reconstruct_rope.default(
        u, sv, positions, inverse_frequencies, out
    )
    return out


def shadowkv_reconstruct(
    u: torch.Tensor,
    sv: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather U and reconstruct 128-element pre-RoPE keys on B200."""

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
    _require_b200(device)
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
    torch.ops.sgl_kernel.shadowkv_reconstruct.default(u, sv, positions, out)
    return out


def shadowkv_plan_device(
    selected_chunk_ids: torch.Tensor,
    selected_lengths: torch.Tensor,
    exact_chunk_ids: torch.Tensor,
    exact_lengths: torch.Tensor,
    temporal_chunk_ids: torch.Tensor,
    temporal_component_validity: torch.Tensor,
    temporal_publication_generations: torch.Tensor,
    temporal_request_generations: torch.Tensor,
    temporal_layout_generations: torch.Tensor,
    row_indices: torch.Tensor,
    row_generations: torch.Tensor,
    plan_slots: torch.Tensor,
    *,
    plan_capacity: int,
    out: ShadowKVDevicePlanOutputs | None = None,
) -> ShadowKVDevicePlan:
    """Launch the compact K/V planner without materializing deferred errors."""

    inputs = (
        ("selected_chunk_ids", selected_chunk_ids, torch.int32, 2),
        ("selected_lengths", selected_lengths, torch.int32, 1),
        ("exact_chunk_ids", exact_chunk_ids, torch.int32, 2),
        ("exact_lengths", exact_lengths, torch.int32, 1),
        ("temporal_chunk_ids", temporal_chunk_ids, torch.int32, 4),
        (
            "temporal_component_validity",
            temporal_component_validity,
            torch.uint8,
            5,
        ),
        (
            "temporal_publication_generations",
            temporal_publication_generations,
            torch.int64,
            5,
        ),
        (
            "temporal_request_generations",
            temporal_request_generations,
            torch.int64,
            1,
        ),
        (
            "temporal_layout_generations",
            temporal_layout_generations,
            torch.int64,
            1,
        ),
        ("row_indices", row_indices, torch.int32, 2),
        ("row_generations", row_generations, torch.int64, 2),
        ("plan_slots", plan_slots, torch.int32, 1),
    )
    for name, tensor, dtype, dimensions in inputs:
        _require_tensor(name, tensor, dtype=dtype, dimensions=dimensions)
    rows, selected_capacity = selected_chunk_ids.shape
    exact_capacity = exact_chunk_ids.shape[1]
    request_slots, local_layers, kv_heads, temporal_capacity = temporal_chunk_ids.shape
    if not 1 <= selected_capacity <= 256:
        raise ValueError("selected capacity must be between 1 and 256")
    if exact_capacity > 64:
        raise ValueError("exact capacity must not exceed 64")
    if request_slots < 1 or local_layers < 1 or kv_heads < 1:
        raise ValueError("temporal owner dimensions must be positive")
    if temporal_capacity > selected_capacity:
        raise ValueError("temporal capacity must not exceed selected capacity")
    if isinstance(plan_capacity, bool) or not isinstance(plan_capacity, int):
        raise TypeError("plan_capacity must be an integer")
    if plan_capacity < 1:
        raise ValueError("plan_capacity must be positive")
    if 2 * plan_capacity * kv_heads * selected_capacity > 2**31:
        raise ValueError("logical destination slots exceed int32")
    if 2 * request_slots * local_layers * kv_heads * temporal_capacity > 2**31:
        raise ValueError("logical source slots exceed int32")
    if (
        selected_lengths.shape != (rows,)
        or exact_chunk_ids.shape[0] != rows
        or exact_lengths.shape != (rows,)
        or plan_slots.shape != (rows,)
    ):
        raise ValueError("selected, exact, and plan-slot row counts differ")
    if row_indices.shape != (rows, 3) or row_generations.shape != (rows, 3):
        raise ValueError("row identity tensors must have shape [rows, 3]")
    temporal_shape = (
        2,
        request_slots,
        local_layers,
        kv_heads,
        temporal_capacity,
    )
    if (
        temporal_component_validity.shape != temporal_shape
        or temporal_publication_generations.shape != temporal_shape
    ):
        raise ValueError("temporal component tensors have incompatible shapes")
    if temporal_request_generations.shape != (
        request_slots,
    ) or temporal_layout_generations.shape != (request_slots,):
        raise ValueError("temporal owner generations must have shape [request_slots]")
    device = selected_chunk_ids.device
    if any(tensor.device != device for _, tensor, _, _ in inputs):
        raise ValueError("all device-plan tensors must share one CUDA device")
    _require_b200(device)

    component_shape = (2, rows, selected_capacity)
    if out is None:
        out = ShadowKVDevicePlanOutputs(
            component_kinds=torch.empty(
                component_shape, dtype=torch.int8, device=device
            ),
            source_slots=torch.empty(component_shape, dtype=torch.int32, device=device),
            destination_slots=torch.empty(
                component_shape, dtype=torch.int32, device=device
            ),
            miss_ordinals=torch.empty(
                component_shape, dtype=torch.int32, device=device
            ),
            counts=torch.empty((2, rows, 2), dtype=torch.int32, device=device),
            error_codes=torch.empty((rows,), dtype=torch.int32, device=device),
        )
    output_specs = (
        ("out.component_kinds", out.component_kinds, torch.int8, 3, component_shape),
        ("out.source_slots", out.source_slots, torch.int32, 3, component_shape),
        (
            "out.destination_slots",
            out.destination_slots,
            torch.int32,
            3,
            component_shape,
        ),
        ("out.miss_ordinals", out.miss_ordinals, torch.int32, 3, component_shape),
        ("out.counts", out.counts, torch.int32, 3, (2, rows, 2)),
        ("out.error_codes", out.error_codes, torch.int32, 1, (rows,)),
    )
    for name, tensor, dtype, dimensions, shape in output_specs:
        _require_tensor(name, tensor, dtype=dtype, dimensions=dimensions)
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if tensor.device != device:
            raise ValueError(f"{name} must share the device-plan CUDA device")
    torch.ops.sgl_kernel.shadowkv_plan_device.default(
        selected_chunk_ids,
        selected_lengths,
        exact_chunk_ids,
        exact_lengths,
        temporal_chunk_ids,
        temporal_component_validity,
        temporal_publication_generations,
        temporal_request_generations,
        temporal_layout_generations,
        row_indices,
        row_generations,
        plan_slots,
        plan_capacity,
        out.component_kinds,
        out.source_slots,
        out.destination_slots,
        out.miss_ordinals,
        out.counts,
        out.error_codes,
    )
    return ShadowKVDevicePlan(
        row_indices=row_indices,
        row_generations=row_generations,
        selected_chunk_ids=selected_chunk_ids,
        selected_lengths=selected_lengths,
        plan_slots=plan_slots,
        component_kinds=out.component_kinds,
        source_slots=out.source_slots,
        destination_slots=out.destination_slots,
        miss_ordinals=out.miss_ordinals,
        counts=out.counts,
        error_codes=out.error_codes,
    )


def shadowkv_place_device(
    component_kinds: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
    plan_slots: torch.Tensor,
    planner_error_codes: torch.Tensor,
    temporal_key_values: torch.Tensor,
    compatibility_key_values: torch.Tensor,
    destination_key_values: torch.Tensor,
    *,
    plan_capacity: int,
) -> torch.Tensor:
    """Place temporal hits and full-work compatibility outputs into one stable slot."""

    inputs = (
        ("component_kinds", component_kinds, torch.int8, 3),
        ("source_slots", source_slots, torch.int32, 3),
        ("destination_slots", destination_slots, torch.int32, 3),
        ("plan_slots", plan_slots, torch.int32, 1),
        ("planner_error_codes", planner_error_codes, torch.int32, 1),
        ("temporal_key_values", temporal_key_values, torch.bfloat16, 7),
        (
            "compatibility_key_values",
            compatibility_key_values,
            torch.bfloat16,
            5,
        ),
        (
            "destination_key_values",
            destination_key_values,
            torch.bfloat16,
            5,
        ),
    )
    for name, tensor, dtype, dimensions in inputs:
        _require_tensor(name, tensor, dtype=dtype, dimensions=dimensions)
    if component_kinds.shape[0] != 2:
        raise ValueError("component_kinds must have shape [2, heads, selected]")
    heads, selected_capacity = component_kinds.shape[1:]
    if not 1 <= selected_capacity <= 256 or heads < 1:
        raise ValueError("placement heads and selected capacity are outside bounds")
    if (
        source_slots.shape != component_kinds.shape
        or destination_slots.shape != component_kinds.shape
    ):
        raise ValueError("placement plan tensors must share shape [2, heads, selected]")
    if plan_slots.shape != (heads,) or planner_error_codes.shape != (heads,):
        raise ValueError("placement row tensors must have shape [heads]")
    if temporal_key_values.shape[0] != 2 or temporal_key_values.shape[3] != heads:
        raise ValueError("temporal K/V component or head dimensions differ")
    if temporal_key_values.shape[-2:] != (8, 128):
        raise ValueError("temporal K/V must use chunk_size=8 and head_dim=128")
    expected_output = (2, heads, selected_capacity, 8, 128)
    if compatibility_key_values.shape != expected_output:
        raise ValueError(f"compatibility_key_values must have shape {expected_output}")
    if destination_key_values.shape != expected_output:
        raise ValueError(f"destination_key_values must have shape {expected_output}")
    if isinstance(plan_capacity, bool) or not isinstance(plan_capacity, int):
        raise TypeError("plan_capacity must be an integer")
    if plan_capacity < 1:
        raise ValueError("plan_capacity must be positive")
    device = component_kinds.device
    if any(tensor.device != device for _, tensor, _, _ in inputs):
        raise ValueError("all placement tensors must share one CUDA device")
    _require_b200(device)
    torch.ops.sgl_kernel.shadowkv_place_device.default(
        component_kinds,
        source_slots,
        destination_slots,
        plan_slots,
        planner_error_codes,
        temporal_key_values,
        compatibility_key_values,
        plan_capacity,
        destination_key_values,
    )
    return destination_key_values


def shadowkv_publish_device(
    selected_chunk_ids: torch.Tensor,
    selected_lengths: torch.Tensor,
    exact_chunk_ids: torch.Tensor,
    exact_lengths: torch.Tensor,
    row_indices: torch.Tensor,
    row_generations: torch.Tensor,
    planner_error_codes: torch.Tensor,
    destination_key_values: torch.Tensor,
    temporal_request_generations: torch.Tensor,
    temporal_layout_generations: torch.Tensor,
    temporal_chunk_ids: torch.Tensor,
    temporal_key_values: torch.Tensor,
    temporal_publication_generations: torch.Tensor,
    temporal_component_validity: torch.Tensor,
) -> None:
    """Publish the effective selected prefix after a caller-ordered completion."""

    inputs = (
        ("selected_chunk_ids", selected_chunk_ids, torch.int32, 2),
        ("selected_lengths", selected_lengths, torch.int32, 1),
        ("exact_chunk_ids", exact_chunk_ids, torch.int32, 2),
        ("exact_lengths", exact_lengths, torch.int32, 1),
        ("row_indices", row_indices, torch.int32, 2),
        ("row_generations", row_generations, torch.int64, 2),
        ("planner_error_codes", planner_error_codes, torch.int32, 1),
        (
            "destination_key_values",
            destination_key_values,
            torch.bfloat16,
            5,
        ),
        (
            "temporal_request_generations",
            temporal_request_generations,
            torch.int64,
            1,
        ),
        (
            "temporal_layout_generations",
            temporal_layout_generations,
            torch.int64,
            1,
        ),
        ("temporal_chunk_ids", temporal_chunk_ids, torch.int32, 4),
        ("temporal_key_values", temporal_key_values, torch.bfloat16, 7),
        (
            "temporal_publication_generations",
            temporal_publication_generations,
            torch.int64,
            5,
        ),
        (
            "temporal_component_validity",
            temporal_component_validity,
            torch.uint8,
            5,
        ),
    )
    for name, tensor, dtype, dimensions in inputs:
        _require_tensor(name, tensor, dtype=dtype, dimensions=dimensions)
    heads, selected_capacity = selected_chunk_ids.shape
    exact_capacity = exact_chunk_ids.shape[1]
    request_slots, local_layers, temporal_heads, temporal_capacity = (
        temporal_chunk_ids.shape
    )
    if heads < 1 or not 1 <= selected_capacity <= 256:
        raise ValueError("publication heads or selected capacity are outside bounds")
    if exact_capacity > 64 or temporal_capacity > selected_capacity:
        raise ValueError("publication exact or temporal capacity is outside bounds")
    if (
        selected_lengths.shape != (heads,)
        or exact_chunk_ids.shape[0] != heads
        or exact_lengths.shape != (heads,)
        or row_indices.shape != (heads, 3)
        or row_generations.shape != (heads, 3)
        or planner_error_codes.shape != (heads,)
    ):
        raise ValueError("publication row tensors have incompatible shapes")
    if temporal_heads != heads:
        raise ValueError("publication temporal and selected head counts differ")
    expected_destination = (2, heads, selected_capacity, 8, 128)
    if destination_key_values.shape != expected_destination:
        raise ValueError(
            f"destination_key_values must have shape {expected_destination}"
        )
    expected_temporal_values = (
        2,
        request_slots,
        local_layers,
        heads,
        temporal_capacity,
        8,
        128,
    )
    if temporal_key_values.shape != expected_temporal_values:
        raise ValueError(
            f"temporal_key_values must have shape {expected_temporal_values}"
        )
    expected_component_metadata = (
        2,
        request_slots,
        local_layers,
        heads,
        temporal_capacity,
    )
    if (
        temporal_publication_generations.shape != expected_component_metadata
        or temporal_component_validity.shape != expected_component_metadata
    ):
        raise ValueError("temporal component metadata shapes differ")
    if temporal_request_generations.shape != (
        request_slots,
    ) or temporal_layout_generations.shape != (request_slots,):
        raise ValueError("temporal owner generations must have shape [request_slots]")
    device = selected_chunk_ids.device
    if any(tensor.device != device for _, tensor, _, _ in inputs):
        raise ValueError("all publication tensors must share one CUDA device")
    _require_b200(device)
    torch.ops.sgl_kernel.shadowkv_publish_device.default(
        selected_chunk_ids,
        selected_lengths,
        exact_chunk_ids,
        exact_lengths,
        row_indices,
        row_generations,
        planner_error_codes,
        destination_key_values,
        temporal_request_generations,
        temporal_layout_generations,
        temporal_chunk_ids,
        temporal_key_values,
        temporal_publication_generations,
        temporal_component_validity,
    )


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
    _require_b200(device)
    plan = torch.full(
        (rows, current_width, 3),
        -1,
        dtype=torch.int64,
        device=current_chunks.device,
    )
    deduplicated_exact = torch.full_like(exact_chunks, -1)
    counts = torch.zeros((rows, 3), dtype=torch.int32, device=current_chunks.device)
    error_codes = torch.zeros((rows,), dtype=torch.int32, device=current_chunks.device)
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
    if validate:
        errors = error_codes.cpu().tolist()
        if any(errors):
            details = ", ".join(
                f"row {row}: code {code}" for row, code in enumerate(errors) if code
            )
            raise ValueError(f"invalid ShadowKV reuse plan input ({details})")
    return ShadowKVReusePlan(plan, deduplicated_exact, counts)
