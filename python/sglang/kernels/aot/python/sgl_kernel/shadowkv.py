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
        "value_miss_chunk_ids",
        "value_miss_lengths",
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
        value_miss_chunk_ids: torch.Tensor | None = None,
        value_miss_lengths: torch.Tensor | None = None,
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
        self.value_miss_chunk_ids = value_miss_chunk_ids
        self.value_miss_lengths = value_miss_lengths


@dataclass(frozen=True)
class ShadowKVDevicePlanOutputs:
    """Caller-owned output tensors for allocation-free device planning."""

    component_kinds: torch.Tensor
    source_slots: torch.Tensor
    destination_slots: torch.Tensor
    miss_ordinals: torch.Tensor
    counts: torch.Tensor
    error_codes: torch.Tensor


@dataclass(frozen=True)
class ShadowKVDevicePlanV2Outputs(ShadowKVDevicePlanOutputs):
    """Caller-owned parallel planner outputs plus compact value misses."""

    value_miss_chunk_ids: torch.Tensor
    value_miss_lengths: torch.Tensor


@dataclass(frozen=True)
class ShadowKVMappedHostRegion:
    """One mapped pinned host tensor whose lifetime protects its device pointer."""

    values: torch.Tensor
    device: torch.device
    device_pointer: int
    byte_length: int

    @property
    def kv_heads(self) -> int:
        return self.values.shape[0]

    @property
    def prompt_chunk_capacity(self) -> int:
        return self.values.shape[1]


def shadowkv_kernels_available() -> bool:
    """Return whether this wheel contains the optional ShadowKV operators."""

    return all(
        hasattr(torch.ops.sgl_kernel, name)
        for name in (
            "shadowkv_reconstruct_generic_aot_v1",
            "shadowkv_reconstruct_rope_generic_aot_v1",
            "shadowkv_plan_reuse_generic_aot_v1",
            "shadowkv_packed_gqa_generic_aot_v1",
            "shadowkv_plan_device_generic_aot_v1",
            "shadowkv_plan_device_v2_generic_aot_v1",
            "shadowkv_publish_value_descriptor_generic_aot_v1",
            "shadowkv_resolve_mapped_host_pointer_generic_aot_v1",
            "shadowkv_place_device_generic_aot_v1",
            "shadowkv_place_device_miss_only_generic_aot_v1",
            "shadowkv_place_device_mapped_host_generic_aot_v1",
            "shadowkv_publish_device_generic_aot_v1",
        )
    )


def shadowkv_a100_fused_key_kernels_available() -> bool:
    """Return whether the optional SM80 fused-key child is installed."""

    return all(
        hasattr(torch.ops.sgl_kernel, name)
        for name in (
            "shadowkv_fused_key_sm80_a100_v2",
            "shadowkv_fused_key_sm80_a100_v3",
            "shadowkv_fused_key_sm80_a100_v4",
            "shadowkv_fused_key_mapped_value_sm80_a100_v3",
            "shadowkv_fused_key_mapped_value_sm80_a100_v4",
            "shadowkv_fused_key_mapped_value_sm80_a100_v5",
            "shadowkv_fused_key_mapped_value_sm80_a100_v6",
            "shadowkv_fused_key_mapped_value_sm80_a100_v7",
            "shadowkv_prepare_exact_miss_gemm_sm80_a100_v1",
            "shadowkv_resolve_miss_count_pointer_sm80_a100_v1",
            "shadowkv_place_value_sm80_a100_v1",
            "shadowkv_place_value_miss_only_sm80_a100_v1",
            "shadowkv_place_value_mapped_host_sm80_a100_v1",
        )
    )


def _launch_shadowkv_packed_gqa(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    lengths: torch.Tensor,
    weights: torch.Tensor,
    out: torch.Tensor,
) -> None:
    torch.ops.sgl_kernel.shadowkv_packed_gqa_generic_aot_v1.default(
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
    torch.ops.sgl_kernel.shadowkv_reconstruct_rope_generic_aot_v1.default(
        u, sv, positions, inverse_frequencies, out
    )


def _launch_shadowkv_reconstruct(
    u: torch.Tensor,
    sv: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
) -> None:
    torch.ops.sgl_kernel.shadowkv_reconstruct_generic_aot_v1.default(
        u, sv, positions, out
    )


def _launch_shadowkv_fused_key_a100(
    u: torch.Tensor,
    sv: torch.Tensor,
    gathered_u: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    component_kinds: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
    miss_ordinals: torch.Tensor,
    selected_chunk_ids: torch.Tensor,
    selected_lengths: torch.Tensor,
    plan_slots: torch.Tensor,
    planner_error_codes: torch.Tensor,
    temporal_key_values: torch.Tensor,
    plan_capacity: int,
    destination_key_values: torch.Tensor,
) -> None:
    torch.ops.sgl_kernel.shadowkv_fused_key_sm80_a100_v3.default(
        u,
        sv,
        gathered_u,
        cosine,
        sine,
        component_kinds,
        source_slots,
        destination_slots,
        miss_ordinals,
        selected_chunk_ids,
        selected_lengths,
        plan_slots,
        planner_error_codes,
        temporal_key_values,
        plan_capacity,
        destination_key_values,
    )


def _prepare_shadowkv_exact_miss_gemm_a100(device_anchor: torch.Tensor) -> None:
    torch.ops.sgl_kernel.shadowkv_prepare_exact_miss_gemm_sm80_a100_v1.default(
        device_anchor
    )


def _resolve_shadowkv_miss_count_pointer_a100(
    host_miss_counts: torch.Tensor,
    device_index: int,
) -> int:
    return int(
        torch.ops.sgl_kernel.shadowkv_resolve_miss_count_pointer_sm80_a100_v1.default(
            host_miss_counts,
            device_index,
        )
    )


def _launch_shadowkv_fused_key_exact_a100(
    u: torch.Tensor,
    sv: torch.Tensor,
    gathered_u: torch.Tensor,
    reconstructed_misses: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    component_kinds: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
    miss_ordinals: torch.Tensor,
    selected_chunk_ids: torch.Tensor,
    selected_lengths: torch.Tensor,
    plan_slots: torch.Tensor,
    planner_error_codes: torch.Tensor,
    temporal_key_values: torch.Tensor,
    host_miss_counts: torch.Tensor,
    mapped_miss_counts: int,
    miss_count_ready_event: int,
    plan_capacity: int,
    destination_key_values: torch.Tensor,
) -> None:
    torch.ops.sgl_kernel.shadowkv_fused_key_sm80_a100_v4.default(
        u,
        sv,
        gathered_u,
        reconstructed_misses,
        cosine,
        sine,
        component_kinds,
        source_slots,
        destination_slots,
        miss_ordinals,
        selected_chunk_ids,
        selected_lengths,
        plan_slots,
        planner_error_codes,
        temporal_key_values,
        host_miss_counts,
        mapped_miss_counts,
        miss_count_ready_event,
        plan_capacity,
        destination_key_values,
    )


def _launch_shadowkv_place_value_a100(
    component_kinds: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
    selected_lengths: torch.Tensor,
    plan_slots: torch.Tensor,
    planner_error_codes: torch.Tensor,
    temporal_key_values: torch.Tensor,
    compatibility_key_values: torch.Tensor,
    plan_capacity: int,
    destination_key_values: torch.Tensor,
) -> None:
    torch.ops.sgl_kernel.shadowkv_place_value_sm80_a100_v1.default(
        component_kinds,
        source_slots,
        destination_slots,
        selected_lengths,
        plan_slots,
        planner_error_codes,
        temporal_key_values,
        compatibility_key_values,
        plan_capacity,
        destination_key_values,
    )


def _launch_shadowkv_place_value_miss_only_a100(
    component_kinds: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
    miss_ordinals: torch.Tensor,
    selected_chunk_ids: torch.Tensor,
    selected_lengths: torch.Tensor,
    plan_slots: torch.Tensor,
    planner_error_codes: torch.Tensor,
    temporal_key_values: torch.Tensor,
    value_miss_chunk_ids: torch.Tensor,
    value_miss_lengths: torch.Tensor,
    descriptor_generation: torch.Tensor,
    descriptor_validity: torch.Tensor,
    expected_generation: int,
    plan_capacity: int,
    value_miss_key_values: torch.Tensor,
    destination_key_values: torch.Tensor,
) -> None:
    torch.ops.sgl_kernel.shadowkv_place_value_miss_only_sm80_a100_v1.default(
        component_kinds,
        source_slots,
        destination_slots,
        miss_ordinals,
        selected_chunk_ids,
        selected_lengths,
        plan_slots,
        planner_error_codes,
        temporal_key_values,
        value_miss_chunk_ids,
        value_miss_lengths,
        descriptor_generation,
        descriptor_validity,
        expected_generation,
        plan_capacity,
        value_miss_key_values,
        destination_key_values,
    )


def _launch_shadowkv_place_value_mapped_host_a100(
    component_kinds: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
    miss_ordinals: torch.Tensor,
    selected_chunk_ids: torch.Tensor,
    selected_lengths: torch.Tensor,
    plan_slots: torch.Tensor,
    planner_error_codes: torch.Tensor,
    temporal_key_values: torch.Tensor,
    value_miss_chunk_ids: torch.Tensor,
    value_miss_lengths: torch.Tensor,
    descriptor_generation: torch.Tensor,
    descriptor_validity: torch.Tensor,
    mapped_host_region: ShadowKVMappedHostRegion,
    prompt_tokens: int,
    expected_generation: int,
    plan_capacity: int,
    destination_key_values: torch.Tensor,
) -> None:
    torch.ops.sgl_kernel.shadowkv_place_value_mapped_host_sm80_a100_v1.default(
        component_kinds,
        source_slots,
        destination_slots,
        miss_ordinals,
        selected_chunk_ids,
        selected_lengths,
        plan_slots,
        planner_error_codes,
        temporal_key_values,
        value_miss_chunk_ids,
        value_miss_lengths,
        descriptor_generation,
        descriptor_validity,
        mapped_host_region.device_pointer,
        mapped_host_region.byte_length,
        mapped_host_region.prompt_chunk_capacity,
        prompt_tokens,
        expected_generation,
        plan_capacity,
        destination_key_values,
    )


def _launch_shadowkv_fused_key_mapped_value_a100(
    u: torch.Tensor,
    sv: torch.Tensor,
    gathered_u: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    component_kinds: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
    miss_ordinals: torch.Tensor,
    selected_chunk_ids: torch.Tensor,
    selected_lengths: torch.Tensor,
    plan_slots: torch.Tensor,
    planner_error_codes: torch.Tensor,
    temporal_key_values: torch.Tensor,
    value_miss_chunk_ids: torch.Tensor,
    value_miss_lengths: torch.Tensor,
    descriptor_generation: torch.Tensor,
    descriptor_validity: torch.Tensor,
    mapped_host_region: ShadowKVMappedHostRegion,
    prompt_tokens: int,
    expected_generation: int,
    plan_capacity: int,
    reconstruction_stream: int,
    destination_key_values: torch.Tensor,
) -> None:
    torch.ops.sgl_kernel.shadowkv_fused_key_mapped_value_sm80_a100_v6.default(
        u,
        sv,
        gathered_u,
        cosine,
        sine,
        component_kinds,
        source_slots,
        destination_slots,
        miss_ordinals,
        selected_chunk_ids,
        selected_lengths,
        plan_slots,
        planner_error_codes,
        temporal_key_values,
        value_miss_chunk_ids,
        value_miss_lengths,
        descriptor_generation,
        descriptor_validity,
        mapped_host_region.device_pointer,
        mapped_host_region.byte_length,
        mapped_host_region.prompt_chunk_capacity,
        prompt_tokens,
        expected_generation,
        plan_capacity,
        reconstruction_stream,
        destination_key_values,
    )


def _launch_shadowkv_fused_key_mapped_value_exact_a100(
    u: torch.Tensor,
    sv: torch.Tensor,
    gathered_u: torch.Tensor,
    reconstructed_misses: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    component_kinds: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
    miss_ordinals: torch.Tensor,
    selected_chunk_ids: torch.Tensor,
    selected_lengths: torch.Tensor,
    plan_slots: torch.Tensor,
    planner_error_codes: torch.Tensor,
    temporal_key_values: torch.Tensor,
    value_miss_chunk_ids: torch.Tensor,
    value_miss_lengths: torch.Tensor,
    descriptor_generation: torch.Tensor,
    descriptor_validity: torch.Tensor,
    mapped_host_region: ShadowKVMappedHostRegion,
    host_miss_counts: torch.Tensor,
    mapped_miss_counts: int,
    miss_count_ready_event: int,
    prompt_tokens: int,
    expected_generation: int,
    plan_capacity: int,
    reconstruction_stream: int,
    destination_key_values: torch.Tensor,
) -> None:
    torch.ops.sgl_kernel.shadowkv_fused_key_mapped_value_sm80_a100_v7.default(
        u,
        sv,
        gathered_u,
        reconstructed_misses,
        cosine,
        sine,
        component_kinds,
        source_slots,
        destination_slots,
        miss_ordinals,
        selected_chunk_ids,
        selected_lengths,
        plan_slots,
        planner_error_codes,
        temporal_key_values,
        value_miss_chunk_ids,
        value_miss_lengths,
        descriptor_generation,
        descriptor_validity,
        mapped_host_region.device_pointer,
        mapped_host_region.byte_length,
        mapped_host_region.prompt_chunk_capacity,
        prompt_tokens,
        expected_generation,
        host_miss_counts,
        mapped_miss_counts,
        miss_count_ready_event,
        plan_capacity,
        reconstruction_stream,
        destination_key_values,
    )


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


def _require_a100_fused_device(device: torch.device) -> None:
    if not shadowkv_a100_fused_key_kernels_available():
        raise RuntimeError(
            "the installed sglang-kernel wheel has no A100 fused-key child"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("A100 fused-key kernels require a visible NVIDIA GPU")
    capability = torch.cuda.get_device_capability(device)
    if capability != (8, 0):
        raise RuntimeError(
            f"A100 fused-key kernels require compute capability 8.0; found {capability}"
        )


def _validate_a100_fused_plan(
    component_kinds: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
    selected_lengths: torch.Tensor,
    plan_slots: torch.Tensor,
    planner_error_codes: torch.Tensor,
    temporal_key_values: torch.Tensor,
    destination_key_values: torch.Tensor,
    *,
    plan_capacity: int,
) -> torch.device:
    specifications = (
        ("component_kinds", component_kinds, torch.int8, 3),
        ("source_slots", source_slots, torch.int32, 3),
        ("destination_slots", destination_slots, torch.int32, 3),
        ("selected_lengths", selected_lengths, torch.int32, 1),
        ("plan_slots", plan_slots, torch.int32, 1),
        ("planner_error_codes", planner_error_codes, torch.int32, 1),
        ("temporal_key_values", temporal_key_values, torch.bfloat16, 7),
        ("destination_key_values", destination_key_values, torch.bfloat16, 5),
    )
    for name, tensor, dtype, dimensions in specifications:
        _require_tensor(name, tensor, dtype=dtype, dimensions=dimensions)
    plan_shape = (2, 8, 256)
    if component_kinds.shape != plan_shape:
        raise ValueError("component_kinds must have shape [2, 8, 256]")
    if source_slots.shape != plan_shape or destination_slots.shape != plan_shape:
        raise ValueError("source and destination slots must have shape [2, 8, 256]")
    if (
        selected_lengths.shape != (8,)
        or plan_slots.shape != (8,)
        or planner_error_codes.shape != (8,)
    ):
        raise ValueError("plan rows must have shape [8]")
    if (
        temporal_key_values.shape[0] != 2
        or temporal_key_values.shape[3] != 8
        or temporal_key_values.shape[5:] != (8, 128)
    ):
        raise ValueError(
            "temporal_key_values must have shape "
            "[2, requests, layers, 8, temporal, 8, 128]"
        )
    if destination_key_values.shape != (2, 8, 256, 8, 128):
        raise ValueError("destination_key_values must have shape [2, 8, 256, 8, 128]")
    if isinstance(plan_capacity, bool) or not isinstance(plan_capacity, int):
        raise TypeError("plan_capacity must be an integer")
    if plan_capacity < 1:
        raise ValueError("plan_capacity must be positive")
    device = component_kinds.device
    if any(tensor.device != device for _, tensor, _, _ in specifications):
        raise ValueError("all A100 fused-plan tensors must share one CUDA device")
    _require_a100_fused_device(device)
    return device


def _validate_a100_fused_key_inputs(
    u: torch.Tensor,
    sv: torch.Tensor,
    gathered_u: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    component_kinds: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
    miss_ordinals: torch.Tensor,
    selected_chunk_ids: torch.Tensor,
    selected_lengths: torch.Tensor,
    plan_slots: torch.Tensor,
    planner_error_codes: torch.Tensor,
    temporal_key_values: torch.Tensor,
    *,
    plan_capacity: int,
    out: torch.Tensor,
) -> torch.device:
    _require_tensor("u", u, dtype=torch.bfloat16, dimensions=2)
    _require_tensor("sv", sv, dtype=torch.bfloat16, dimensions=3)
    _require_tensor("gathered_u", gathered_u, dtype=torch.bfloat16, dimensions=3)
    _require_tensor("cosine", cosine, dtype=torch.float32, dimensions=2)
    _require_tensor("sine", sine, dtype=torch.float32, dimensions=2)
    _require_tensor("miss_ordinals", miss_ordinals, dtype=torch.int32, dimensions=3)
    _require_tensor(
        "selected_chunk_ids", selected_chunk_ids, dtype=torch.int32, dimensions=2
    )
    device = _validate_a100_fused_plan(
        component_kinds,
        source_slots,
        destination_slots,
        selected_lengths,
        plan_slots,
        planner_error_codes,
        temporal_key_values,
        out,
        plan_capacity=plan_capacity,
    )
    if not 1 <= u.shape[0] <= 8192 or u.shape[1] != 160:
        raise ValueError("u must have shape [1..8192, 160]")
    if sv.shape != (8, 160, 128):
        raise ValueError("sv must have shape [8, 160, 128]")
    if gathered_u.shape != (8, 2048, 160):
        raise ValueError("gathered_u must have shape [8, 2048, 160]")
    if (
        cosine.shape[0] < u.shape[0]
        or cosine.shape[0] > 8192
        or cosine.shape[1] != 64
        or sine.shape != cosine.shape
    ):
        raise ValueError("cosine and sine must share shape [u_tokens..8192, 64]")
    if miss_ordinals.shape != (2, 8, 256):
        raise ValueError("miss_ordinals must have shape [2, 8, 256]")
    if selected_chunk_ids.shape != (8, 256):
        raise ValueError("selected_chunk_ids must have shape [8, 256]")
    if any(
        tensor.device != device
        for tensor in (
            u,
            sv,
            gathered_u,
            cosine,
            sine,
            miss_ordinals,
            selected_chunk_ids,
        )
    ):
        raise ValueError("all A100 fused-key tensors must share one CUDA device")
    return device


def shadowkv_fused_key_a100(
    u: torch.Tensor,
    sv: torch.Tensor,
    gathered_u: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    component_kinds: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
    miss_ordinals: torch.Tensor,
    selected_chunk_ids: torch.Tensor,
    selected_lengths: torch.Tensor,
    plan_slots: torch.Tensor,
    planner_error_codes: torch.Tensor,
    temporal_key_values: torch.Tensor,
    *,
    plan_capacity: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Place K hits and reconstruct K misses directly into an A100 plan slot."""

    _validate_a100_fused_key_inputs(
        u,
        sv,
        gathered_u,
        cosine,
        sine,
        component_kinds,
        source_slots,
        destination_slots,
        miss_ordinals,
        selected_chunk_ids,
        selected_lengths,
        plan_slots,
        planner_error_codes,
        temporal_key_values,
        plan_capacity=plan_capacity,
        out=out,
    )
    _launch_shadowkv_fused_key_a100(
        u,
        sv,
        gathered_u,
        cosine,
        sine,
        component_kinds,
        source_slots,
        destination_slots,
        miss_ordinals,
        selected_chunk_ids,
        selected_lengths,
        plan_slots,
        planner_error_codes,
        temporal_key_values,
        plan_capacity,
        out,
    )
    return out[0].view(8, 2048, 128)


def shadowkv_place_value_a100(
    component_kinds: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
    selected_lengths: torch.Tensor,
    plan_slots: torch.Tensor,
    planner_error_codes: torch.Tensor,
    temporal_key_values: torch.Tensor,
    compatibility_key_values: torch.Tensor,
    *,
    plan_capacity: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Place compatibility V hits and misses without touching fused K output."""

    device = _validate_a100_fused_plan(
        component_kinds,
        source_slots,
        destination_slots,
        selected_lengths,
        plan_slots,
        planner_error_codes,
        temporal_key_values,
        out,
        plan_capacity=plan_capacity,
    )
    _require_tensor(
        "compatibility_key_values",
        compatibility_key_values,
        dtype=torch.bfloat16,
        dimensions=5,
    )
    if compatibility_key_values.shape != (2, 8, 256, 8, 128):
        raise ValueError("compatibility_key_values must have shape [2, 8, 256, 8, 128]")
    if compatibility_key_values.device != device:
        raise ValueError("compatibility values must share the plan CUDA device")
    _launch_shadowkv_place_value_a100(
        component_kinds,
        source_slots,
        destination_slots,
        selected_lengths,
        plan_slots,
        planner_error_codes,
        temporal_key_values,
        compatibility_key_values,
        plan_capacity,
        out,
    )
    return out[1].view(8, 2048, 128)


def _validate_a100_value_descriptor(
    miss_ordinals: torch.Tensor,
    selected_chunk_ids: torch.Tensor,
    value_miss_chunk_ids: torch.Tensor,
    value_miss_lengths: torch.Tensor,
    descriptor_generation: torch.Tensor,
    descriptor_validity: torch.Tensor,
    *,
    device: torch.device,
    expected_generation: int,
) -> None:
    specifications = (
        ("miss_ordinals", miss_ordinals, torch.int32, 3, (2, 8, 256)),
        ("selected_chunk_ids", selected_chunk_ids, torch.int32, 2, (8, 256)),
        ("value_miss_chunk_ids", value_miss_chunk_ids, torch.int32, 2, (8, 256)),
        ("value_miss_lengths", value_miss_lengths, torch.int32, 1, (8,)),
        ("descriptor_generation", descriptor_generation, torch.int64, 1, (1,)),
        ("descriptor_validity", descriptor_validity, torch.uint8, 1, (1,)),
    )
    for name, tensor, dtype, dimensions, shape in specifications:
        _require_tensor(name, tensor, dtype=dtype, dimensions=dimensions)
        if tensor.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if tensor.device != device:
            raise ValueError(f"{name} must share the plan CUDA device")
    if isinstance(expected_generation, bool) or not isinstance(
        expected_generation, int
    ):
        raise TypeError("expected_generation must be an integer")
    if expected_generation < 0:
        raise ValueError("expected_generation must be nonnegative")


def _validate_a100_mapped_value_inputs(
    component_kinds: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
    miss_ordinals: torch.Tensor,
    selected_chunk_ids: torch.Tensor,
    selected_lengths: torch.Tensor,
    plan_slots: torch.Tensor,
    planner_error_codes: torch.Tensor,
    temporal_key_values: torch.Tensor,
    value_miss_chunk_ids: torch.Tensor,
    value_miss_lengths: torch.Tensor,
    descriptor_generation: torch.Tensor,
    descriptor_validity: torch.Tensor,
    mapped_host_region: ShadowKVMappedHostRegion,
    *,
    prompt_tokens: int,
    expected_generation: int,
    plan_capacity: int,
    out: torch.Tensor,
) -> torch.device:
    device = _validate_a100_fused_plan(
        component_kinds,
        source_slots,
        destination_slots,
        selected_lengths,
        plan_slots,
        planner_error_codes,
        temporal_key_values,
        out,
        plan_capacity=plan_capacity,
    )
    _validate_a100_value_descriptor(
        miss_ordinals,
        selected_chunk_ids,
        value_miss_chunk_ids,
        value_miss_lengths,
        descriptor_generation,
        descriptor_validity,
        device=device,
        expected_generation=expected_generation,
    )
    if not isinstance(mapped_host_region, ShadowKVMappedHostRegion):
        raise TypeError("mapped_host_region must be a ShadowKVMappedHostRegion")
    if mapped_host_region.device != device:
        raise ValueError("mapped host region must target the plan CUDA device")
    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int):
        raise TypeError("prompt_tokens must be an integer")
    if not 1 <= prompt_tokens <= mapped_host_region.prompt_chunk_capacity * 8:
        raise ValueError("prompt_tokens exceed the mapped host region")
    return device


def shadowkv_place_value_miss_only_a100(
    component_kinds: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
    miss_ordinals: torch.Tensor,
    selected_chunk_ids: torch.Tensor,
    selected_lengths: torch.Tensor,
    plan_slots: torch.Tensor,
    planner_error_codes: torch.Tensor,
    temporal_key_values: torch.Tensor,
    value_miss_chunk_ids: torch.Tensor,
    value_miss_lengths: torch.Tensor,
    descriptor_generation: torch.Tensor,
    descriptor_validity: torch.Tensor,
    value_miss_key_values: torch.Tensor,
    *,
    expected_generation: int,
    plan_capacity: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Place compact V misses and temporal V hits beside completed fused K."""

    device = _validate_a100_fused_plan(
        component_kinds,
        source_slots,
        destination_slots,
        selected_lengths,
        plan_slots,
        planner_error_codes,
        temporal_key_values,
        out,
        plan_capacity=plan_capacity,
    )
    _validate_a100_value_descriptor(
        miss_ordinals,
        selected_chunk_ids,
        value_miss_chunk_ids,
        value_miss_lengths,
        descriptor_generation,
        descriptor_validity,
        device=device,
        expected_generation=expected_generation,
    )
    _require_tensor(
        "value_miss_key_values",
        value_miss_key_values,
        dtype=torch.bfloat16,
        dimensions=4,
    )
    if value_miss_key_values.shape != (8, 256, 8, 128):
        raise ValueError("value_miss_key_values must have shape [8, 256, 8, 128]")
    if value_miss_key_values.device != device:
        raise ValueError("value misses must share the plan CUDA device")
    _launch_shadowkv_place_value_miss_only_a100(
        component_kinds,
        source_slots,
        destination_slots,
        miss_ordinals,
        selected_chunk_ids,
        selected_lengths,
        plan_slots,
        planner_error_codes,
        temporal_key_values,
        value_miss_chunk_ids,
        value_miss_lengths,
        descriptor_generation,
        descriptor_validity,
        expected_generation,
        plan_capacity,
        value_miss_key_values,
        out,
    )
    return out[1].view(8, 2048, 128)


def shadowkv_place_value_mapped_host_a100(
    component_kinds: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
    miss_ordinals: torch.Tensor,
    selected_chunk_ids: torch.Tensor,
    selected_lengths: torch.Tensor,
    plan_slots: torch.Tensor,
    planner_error_codes: torch.Tensor,
    temporal_key_values: torch.Tensor,
    value_miss_chunk_ids: torch.Tensor,
    value_miss_lengths: torch.Tensor,
    descriptor_generation: torch.Tensor,
    descriptor_validity: torch.Tensor,
    mapped_host_region: ShadowKVMappedHostRegion,
    *,
    prompt_tokens: int,
    expected_generation: int,
    plan_capacity: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Place mapped-host V misses and temporal V hits beside completed fused K."""

    _validate_a100_mapped_value_inputs(
        component_kinds,
        source_slots,
        destination_slots,
        miss_ordinals,
        selected_chunk_ids,
        selected_lengths,
        plan_slots,
        planner_error_codes,
        temporal_key_values,
        value_miss_chunk_ids,
        value_miss_lengths,
        descriptor_generation,
        descriptor_validity,
        mapped_host_region,
        prompt_tokens=prompt_tokens,
        expected_generation=expected_generation,
        plan_capacity=plan_capacity,
        out=out,
    )
    _launch_shadowkv_place_value_mapped_host_a100(
        component_kinds,
        source_slots,
        destination_slots,
        miss_ordinals,
        selected_chunk_ids,
        selected_lengths,
        plan_slots,
        planner_error_codes,
        temporal_key_values,
        value_miss_chunk_ids,
        value_miss_lengths,
        descriptor_generation,
        descriptor_validity,
        mapped_host_region,
        prompt_tokens,
        expected_generation,
        plan_capacity,
        out,
    )
    return out[1].view(8, 2048, 128)


def shadowkv_fused_key_mapped_value_a100(
    u: torch.Tensor,
    sv: torch.Tensor,
    gathered_u: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    component_kinds: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
    miss_ordinals: torch.Tensor,
    selected_chunk_ids: torch.Tensor,
    selected_lengths: torch.Tensor,
    plan_slots: torch.Tensor,
    planner_error_codes: torch.Tensor,
    temporal_key_values: torch.Tensor,
    value_miss_chunk_ids: torch.Tensor,
    value_miss_lengths: torch.Tensor,
    descriptor_generation: torch.Tensor,
    descriptor_validity: torch.Tensor,
    mapped_host_region: ShadowKVMappedHostRegion,
    reconstruction_stream: torch.cuda.Stream,
    *,
    prompt_tokens: int,
    expected_generation: int,
    plan_capacity: int,
    out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch key prepare, throttled mapped V, then BMM/finalize on A100."""

    device = _validate_a100_fused_key_inputs(
        u,
        sv,
        gathered_u,
        cosine,
        sine,
        component_kinds,
        source_slots,
        destination_slots,
        miss_ordinals,
        selected_chunk_ids,
        selected_lengths,
        plan_slots,
        planner_error_codes,
        temporal_key_values,
        plan_capacity=plan_capacity,
        out=out,
    )
    _validate_a100_mapped_value_inputs(
        component_kinds,
        source_slots,
        destination_slots,
        miss_ordinals,
        selected_chunk_ids,
        selected_lengths,
        plan_slots,
        planner_error_codes,
        temporal_key_values,
        value_miss_chunk_ids,
        value_miss_lengths,
        descriptor_generation,
        descriptor_validity,
        mapped_host_region,
        prompt_tokens=prompt_tokens,
        expected_generation=expected_generation,
        plan_capacity=plan_capacity,
        out=out,
    )
    if not isinstance(reconstruction_stream, torch.cuda.Stream):
        raise TypeError("reconstruction_stream must be a torch.cuda.Stream")
    if reconstruction_stream.device != device:
        raise ValueError("reconstruction_stream must target the plan CUDA device")
    reconstruction_stream_id = int(reconstruction_stream.cuda_stream)
    if reconstruction_stream_id == int(torch.cuda.current_stream(device).cuda_stream):
        raise ValueError("reconstruction and mapped-value streams must be distinct")
    _launch_shadowkv_fused_key_mapped_value_a100(
        u,
        sv,
        gathered_u,
        cosine,
        sine,
        component_kinds,
        source_slots,
        destination_slots,
        miss_ordinals,
        selected_chunk_ids,
        selected_lengths,
        plan_slots,
        planner_error_codes,
        temporal_key_values,
        value_miss_chunk_ids,
        value_miss_lengths,
        descriptor_generation,
        descriptor_validity,
        mapped_host_region,
        prompt_tokens,
        expected_generation,
        plan_capacity,
        reconstruction_stream_id,
        out,
    )
    return (
        out[0].view(8, 2048, 128),
        out[1].view(8, 2048, 128),
    )


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


def _validate_shadowkv_plan_inputs(
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
) -> tuple[int, int, torch.device]:
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
    _require_supported_device(device)
    return rows, selected_capacity, device


def _validate_shadowkv_plan_outputs(
    out: ShadowKVDevicePlanOutputs,
    *,
    rows: int,
    selected_capacity: int,
    device: torch.device,
) -> None:
    component_shape = (2, rows, selected_capacity)
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

    rows, selected_capacity, device = _validate_shadowkv_plan_inputs(
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
        plan_capacity=plan_capacity,
    )

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
    _validate_shadowkv_plan_outputs(
        out,
        rows=rows,
        selected_capacity=selected_capacity,
        device=device,
    )
    torch.ops.sgl_kernel.shadowkv_plan_device_generic_aot_v1.default(
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


def shadowkv_plan_device_v2(
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
    out: ShadowKVDevicePlanV2Outputs | None = None,
) -> ShadowKVDevicePlan:
    """Launch the parallel caller-buffered planner and compact V misses."""

    rows, selected_capacity, device = _validate_shadowkv_plan_inputs(
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
        plan_capacity=plan_capacity,
    )
    component_shape = (2, rows, selected_capacity)
    if out is None:
        out = ShadowKVDevicePlanV2Outputs(
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
            value_miss_chunk_ids=torch.empty(
                (rows, selected_capacity), dtype=torch.int32, device=device
            ),
            value_miss_lengths=torch.empty((rows,), dtype=torch.int32, device=device),
        )
    _validate_shadowkv_plan_outputs(
        out,
        rows=rows,
        selected_capacity=selected_capacity,
        device=device,
    )
    descriptor_specs = (
        (
            "out.value_miss_chunk_ids",
            out.value_miss_chunk_ids,
            2,
            (rows, selected_capacity),
        ),
        ("out.value_miss_lengths", out.value_miss_lengths, 1, (rows,)),
    )
    for name, tensor, dimensions, shape in descriptor_specs:
        _require_tensor(name, tensor, dtype=torch.int32, dimensions=dimensions)
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if tensor.device != device:
            raise ValueError(f"{name} must share the device-plan CUDA device")
    torch.ops.sgl_kernel.shadowkv_plan_device_v2_generic_aot_v1.default(
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
        out.value_miss_chunk_ids,
        out.value_miss_lengths,
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
        value_miss_chunk_ids=out.value_miss_chunk_ids,
        value_miss_lengths=out.value_miss_lengths,
    )


def shadowkv_publish_value_descriptor(
    descriptor_generation: torch.Tensor,
    descriptor_validity: torch.Tensor,
    *,
    generation: int,
) -> None:
    """Publish one compact descriptor generation and set validity last."""

    _require_tensor(
        "descriptor_generation", descriptor_generation, dtype=torch.int64, dimensions=1
    )
    _require_tensor(
        "descriptor_validity", descriptor_validity, dtype=torch.uint8, dimensions=1
    )
    if descriptor_generation.shape != (1,) or descriptor_validity.shape != (1,):
        raise ValueError("descriptor generation and validity must have shape [1]")
    if descriptor_generation.device != descriptor_validity.device:
        raise ValueError(
            "descriptor generation and validity must share one CUDA device"
        )
    if isinstance(generation, bool) or not isinstance(generation, int):
        raise TypeError("generation must be an integer")
    if generation < 0:
        raise ValueError("generation must be nonnegative")
    _require_supported_device(descriptor_generation.device)
    torch.ops.sgl_kernel.shadowkv_publish_value_descriptor_generic_aot_v1.default(
        descriptor_generation,
        descriptor_validity,
        generation,
    )


def shadowkv_resolve_mapped_host_region(
    host_values: torch.Tensor,
    *,
    device: torch.device | str,
) -> ShadowKVMappedHostRegion:
    """Resolve one B200-accessible pinned host region and retain its owner."""

    if host_values.dtype != torch.bfloat16:
        raise ValueError("mapped host values must use torch.bfloat16")
    if host_values.ndim != 4 or host_values.shape[-2:] != (8, 128):
        raise ValueError("mapped host values must have shape [heads, chunks, 8, 128]")
    if host_values.shape[0] < 1 or host_values.shape[1] < 1:
        raise ValueError("mapped host values must contain heads and chunks")
    if host_values.device.type != "cpu":
        raise ValueError("mapped host values must reside on CPU")
    if not host_values.is_contiguous():
        raise ValueError("mapped host values must be contiguous")
    if not host_values.is_pinned():
        raise ValueError("mapped host values must use page-locked memory")
    if host_values.data_ptr() % 16:
        raise ValueError("mapped host values must be 16-byte aligned")
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        raise ValueError("mapped host values require a CUDA destination device")
    if resolved_device.index is None:
        resolved_device = torch.device("cuda", torch.cuda.current_device())
    _require_supported_device(resolved_device)
    pointer = int(
        torch.ops.sgl_kernel.shadowkv_resolve_mapped_host_pointer_generic_aot_v1.default(
            host_values,
            resolved_device.index,
        )
    )
    if pointer <= 0 or pointer % 16:
        raise RuntimeError("mapped host resolver returned an invalid device pointer")
    return ShadowKVMappedHostRegion(
        values=host_values,
        device=resolved_device,
        device_pointer=pointer,
        byte_length=host_values.numel() * host_values.element_size(),
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
    _require_supported_device(device)
    torch.ops.sgl_kernel.shadowkv_place_device_generic_aot_v1.default(
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


def shadowkv_place_device_miss_only(
    component_kinds: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
    miss_ordinals: torch.Tensor,
    selected_chunk_ids: torch.Tensor,
    plan_slots: torch.Tensor,
    planner_error_codes: torch.Tensor,
    temporal_key_values: torch.Tensor,
    reconstructed_keys: torch.Tensor,
    value_miss_chunk_ids: torch.Tensor,
    value_miss_lengths: torch.Tensor,
    descriptor_generation: torch.Tensor,
    descriptor_validity: torch.Tensor,
    value_miss_key_values: torch.Tensor,
    destination_key_values: torch.Tensor,
    *,
    expected_generation: int,
    plan_capacity: int,
) -> torch.Tensor:
    """Place D2D hits, reconstructed K misses, and compact V misses directly."""

    inputs = (
        ("component_kinds", component_kinds, torch.int8, 3),
        ("source_slots", source_slots, torch.int32, 3),
        ("destination_slots", destination_slots, torch.int32, 3),
        ("miss_ordinals", miss_ordinals, torch.int32, 3),
        ("selected_chunk_ids", selected_chunk_ids, torch.int32, 2),
        ("plan_slots", plan_slots, torch.int32, 1),
        ("planner_error_codes", planner_error_codes, torch.int32, 1),
        ("temporal_key_values", temporal_key_values, torch.bfloat16, 7),
        ("reconstructed_keys", reconstructed_keys, torch.bfloat16, 4),
        ("value_miss_chunk_ids", value_miss_chunk_ids, torch.int32, 2),
        ("value_miss_lengths", value_miss_lengths, torch.int32, 1),
        ("descriptor_generation", descriptor_generation, torch.int64, 1),
        ("descriptor_validity", descriptor_validity, torch.uint8, 1),
        ("value_miss_key_values", value_miss_key_values, torch.bfloat16, 4),
        ("destination_key_values", destination_key_values, torch.bfloat16, 5),
    )
    for name, tensor, dtype, dimensions in inputs:
        _require_tensor(name, tensor, dtype=dtype, dimensions=dimensions)
    if component_kinds.shape[0] != 2:
        raise ValueError("component_kinds must have shape [2, heads, selected]")
    heads, selected_capacity = component_kinds.shape[1:]
    component_shape = (2, heads, selected_capacity)
    compact_shape = (heads, selected_capacity)
    if not 1 <= selected_capacity <= 256 or heads < 1:
        raise ValueError("placement heads and selected capacity are outside bounds")
    if any(
        tensor.shape != component_shape
        for tensor in (source_slots, destination_slots, miss_ordinals)
    ):
        raise ValueError("placement plan tensors must share shape [2, heads, selected]")
    if (
        selected_chunk_ids.shape != compact_shape
        or value_miss_chunk_ids.shape != compact_shape
    ):
        raise ValueError(
            "selected and value-miss ids must have shape [heads, selected]"
        )
    if (
        plan_slots.shape != (heads,)
        or planner_error_codes.shape != (heads,)
        or value_miss_lengths.shape != (heads,)
    ):
        raise ValueError("placement row tensors must have shape [heads]")
    if descriptor_generation.shape != (1,) or descriptor_validity.shape != (1,):
        raise ValueError("descriptor generation and validity must have shape [1]")
    if temporal_key_values.shape[0] != 2 or temporal_key_values.shape[3] != heads:
        raise ValueError("temporal K/V component or head dimensions differ")
    if temporal_key_values.shape[-2:] != (8, 128):
        raise ValueError("temporal K/V must use chunk_size=8 and head_dim=128")
    expected_compact_values = (heads, selected_capacity, 8, 128)
    if reconstructed_keys.shape != expected_compact_values:
        raise ValueError(
            f"reconstructed_keys must have shape {expected_compact_values}"
        )
    if value_miss_key_values.shape != expected_compact_values:
        raise ValueError(
            f"value_miss_key_values must have shape {expected_compact_values}"
        )
    expected_output = (2, heads, selected_capacity, 8, 128)
    if destination_key_values.shape != expected_output:
        raise ValueError(f"destination_key_values must have shape {expected_output}")
    if isinstance(expected_generation, bool) or not isinstance(
        expected_generation, int
    ):
        raise TypeError("expected_generation must be an integer")
    if expected_generation < 0:
        raise ValueError("expected_generation must be nonnegative")
    if isinstance(plan_capacity, bool) or not isinstance(plan_capacity, int):
        raise TypeError("plan_capacity must be an integer")
    if plan_capacity < 1:
        raise ValueError("plan_capacity must be positive")
    device = component_kinds.device
    if any(tensor.device != device for _, tensor, _, _ in inputs):
        raise ValueError("all miss-only placement tensors must share one CUDA device")
    _require_supported_device(device)
    torch.ops.sgl_kernel.shadowkv_place_device_miss_only_generic_aot_v1.default(
        component_kinds,
        source_slots,
        destination_slots,
        miss_ordinals,
        selected_chunk_ids,
        plan_slots,
        planner_error_codes,
        temporal_key_values,
        reconstructed_keys,
        value_miss_chunk_ids,
        value_miss_lengths,
        descriptor_generation,
        descriptor_validity,
        expected_generation,
        plan_capacity,
        value_miss_key_values,
        destination_key_values,
    )
    return destination_key_values


def shadowkv_place_device_mapped_host(
    component_kinds: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
    miss_ordinals: torch.Tensor,
    selected_chunk_ids: torch.Tensor,
    plan_slots: torch.Tensor,
    planner_error_codes: torch.Tensor,
    temporal_key_values: torch.Tensor,
    reconstructed_keys: torch.Tensor,
    value_miss_chunk_ids: torch.Tensor,
    value_miss_lengths: torch.Tensor,
    descriptor_generation: torch.Tensor,
    descriptor_validity: torch.Tensor,
    mapped_host_region: ShadowKVMappedHostRegion,
    destination_key_values: torch.Tensor,
    *,
    prompt_tokens: int,
    expected_generation: int,
    plan_capacity: int,
) -> torch.Tensor:
    """Place hits and K misses while reading V misses from mapped pinned host rows."""

    inputs = (
        ("component_kinds", component_kinds, torch.int8, 3),
        ("source_slots", source_slots, torch.int32, 3),
        ("destination_slots", destination_slots, torch.int32, 3),
        ("miss_ordinals", miss_ordinals, torch.int32, 3),
        ("selected_chunk_ids", selected_chunk_ids, torch.int32, 2),
        ("plan_slots", plan_slots, torch.int32, 1),
        ("planner_error_codes", planner_error_codes, torch.int32, 1),
        ("temporal_key_values", temporal_key_values, torch.bfloat16, 7),
        ("reconstructed_keys", reconstructed_keys, torch.bfloat16, 4),
        ("value_miss_chunk_ids", value_miss_chunk_ids, torch.int32, 2),
        ("value_miss_lengths", value_miss_lengths, torch.int32, 1),
        ("descriptor_generation", descriptor_generation, torch.int64, 1),
        ("descriptor_validity", descriptor_validity, torch.uint8, 1),
        ("destination_key_values", destination_key_values, torch.bfloat16, 5),
    )
    for name, tensor, dtype, dimensions in inputs:
        _require_tensor(name, tensor, dtype=dtype, dimensions=dimensions)
    if not isinstance(mapped_host_region, ShadowKVMappedHostRegion):
        raise TypeError("mapped_host_region must be a resolved ShadowKV region")
    if component_kinds.shape[0] != 2:
        raise ValueError("component_kinds must have shape [2, heads, selected]")
    heads, selected_capacity = component_kinds.shape[1:]
    component_shape = (2, heads, selected_capacity)
    compact_shape = (heads, selected_capacity)
    if not 1 <= selected_capacity <= 256 or heads < 1:
        raise ValueError("placement heads and selected capacity are outside bounds")
    if any(
        tensor.shape != component_shape
        for tensor in (source_slots, destination_slots, miss_ordinals)
    ):
        raise ValueError("placement plan tensors must share shape [2, heads, selected]")
    if (
        selected_chunk_ids.shape != compact_shape
        or value_miss_chunk_ids.shape != compact_shape
    ):
        raise ValueError(
            "selected and value-miss ids must have shape [heads, selected]"
        )
    if (
        plan_slots.shape != (heads,)
        or planner_error_codes.shape != (heads,)
        or value_miss_lengths.shape != (heads,)
    ):
        raise ValueError("placement row tensors must have shape [heads]")
    if descriptor_generation.shape != (1,) or descriptor_validity.shape != (1,):
        raise ValueError("descriptor generation and validity must have shape [1]")
    if temporal_key_values.shape[0] != 2 or temporal_key_values.shape[3] != heads:
        raise ValueError("temporal K/V component or head dimensions differ")
    if temporal_key_values.shape[-2:] != (8, 128):
        raise ValueError("temporal K/V must use chunk_size=8 and head_dim=128")
    expected_compact_values = (heads, selected_capacity, 8, 128)
    if reconstructed_keys.shape != expected_compact_values:
        raise ValueError(
            f"reconstructed_keys must have shape {expected_compact_values}"
        )
    expected_output = (2, heads, selected_capacity, 8, 128)
    if destination_key_values.shape != expected_output:
        raise ValueError(f"destination_key_values must have shape {expected_output}")
    if mapped_host_region.kv_heads != heads:
        raise ValueError("mapped host region has another KV-head count")
    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int):
        raise TypeError("prompt_tokens must be an integer")
    if not 1 <= prompt_tokens <= mapped_host_region.prompt_chunk_capacity * 8:
        raise ValueError("prompt_tokens exceeds the mapped host region")
    if isinstance(expected_generation, bool) or not isinstance(
        expected_generation, int
    ):
        raise TypeError("expected_generation must be an integer")
    if expected_generation < 0:
        raise ValueError("expected_generation must be nonnegative")
    if isinstance(plan_capacity, bool) or not isinstance(plan_capacity, int):
        raise TypeError("plan_capacity must be an integer")
    if plan_capacity < 1:
        raise ValueError("plan_capacity must be positive")
    device = component_kinds.device
    if any(tensor.device != device for _, tensor, _, _ in inputs):
        raise ValueError("all mapped-host placement tensors must share one CUDA device")
    if mapped_host_region.device != device:
        raise ValueError("mapped host region belongs to another CUDA device")
    _require_supported_device(device)
    torch.ops.sgl_kernel.shadowkv_place_device_mapped_host_generic_aot_v1.default(
        component_kinds,
        source_slots,
        destination_slots,
        miss_ordinals,
        selected_chunk_ids,
        plan_slots,
        planner_error_codes,
        temporal_key_values,
        reconstructed_keys,
        value_miss_chunk_ids,
        value_miss_lengths,
        descriptor_generation,
        descriptor_validity,
        mapped_host_region.device_pointer,
        mapped_host_region.byte_length,
        mapped_host_region.prompt_chunk_capacity,
        prompt_tokens,
        expected_generation,
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
    _require_supported_device(device)
    torch.ops.sgl_kernel.shadowkv_publish_device_generic_aot_v1.default(
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
