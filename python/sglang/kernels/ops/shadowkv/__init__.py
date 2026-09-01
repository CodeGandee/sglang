"""Stable ShadowKV operations with architecture-aware implementation metadata.

The public functions default to the readable oracle. Production startup uses
the registry selector, prepares the chosen provider, and passes or binds its
callable once in an immutable kernel plan.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final

from sglang.kernels.ops.shadowkv.contracts import (
    ShadowKVReusePlan,
    validate_packed_gqa,
    validate_plan_reuse,
    validate_reconstruct,
    validate_reconstruct_rope,
)
from sglang.kernels.ops.shadowkv.dispatch import (
    bind_implementation,
    prepare_implementation,
    resolve_callable,
)
from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import (
    CapabilityRequirement as Cap,
)
from sglang.kernels.spec import (
    KernelBackend,
    KernelExecutionProperties,
    KernelInputEnvelope,
    KernelSpec,
    KernelSpecialization,
)

RECONSTRUCT: Final = "shadowkv.reconstruct"
RECONSTRUCT_ROPE: Final = "shadowkv.reconstruct_rope"
PLAN_REUSE: Final = "shadowkv.plan_reuse"
PACKED_GQA: Final = "shadowkv.packed_gqa"

REFERENCE_IMPLEMENTATIONS: Final = {
    RECONSTRUCT: "shadowkv.reconstruct.reference.v1",
    RECONSTRUCT_ROPE: "shadowkv.reconstruct-rope.reference.v1",
    PLAN_REUSE: "shadowkv.plan-reuse.reference.v1",
    PACKED_GQA: "shadowkv.packed-gqa.reference.v1",
}
GENERIC_AOT_IMPLEMENTATIONS: Final = {
    RECONSTRUCT: "shadowkv.reconstruct.generic-aot.v1",
    RECONSTRUCT_ROPE: "shadowkv.reconstruct-rope.generic-aot.v1",
    PLAN_REUSE: "shadowkv.plan-reuse.generic-aot.v1",
    PACKED_GQA: "shadowkv.packed-gqa.generic-aot.v1",
}

_PROVIDER_ROOT = "sglang.kernels.ops.shadowkv.providers"
_QUALIFICATION_REFERENCES = (
    "qualification://predkv/shadowkv-a100-kernel-bringup/20260831-baseline",
    "qualification://predkv/b200-shadowkv-operation-baseline/20260831",
)
_AOT_EXECUTION = KernelExecutionProperties(
    deterministic=True,
    current_stream=True,
    graph_compatible=False,
    supports_eager=True,
    workspace_description="caller-owned output and operation-specific scratch",
)
_AOT_GRAPH_EXECUTION = KernelExecutionProperties(
    deterministic=True,
    current_stream=True,
    graph_compatible=True,
    supports_eager=True,
    workspace_description="caller-owned graph-stable output and scratch",
)
_REFERENCE_EXECUTION = KernelExecutionProperties(
    deterministic=True,
    current_stream=True,
    graph_compatible=False,
    supports_eager=True,
)
_ENVELOPES = {
    RECONSTRUCT: KernelInputEnvelope(
        dtypes=frozenset({"bfloat16"}),
        head_dimensions=frozenset({128}),
        factor_ranks=frozenset({64, 128, 160, 256}),
        features=frozenset({"current-stream", "caller-output-buffer"}),
        description="selected-row pre-RoPE reconstruction",
    ),
    RECONSTRUCT_ROPE: KernelInputEnvelope(
        dtypes=frozenset({"bfloat16"}),
        head_dimensions=frozenset({64}),
        factor_ranks=frozenset({160}),
        features=frozenset(
            {"current-stream", "caller-output-buffer", "neox-llama-rope"}
        ),
        description="selected-row reconstruction plus NeoX Llama RoPE",
    ),
    PLAN_REUSE: KernelInputEnvelope(
        dtypes=frozenset({"int64", "int32"}),
        features=frozenset(
            {
                "stable-order",
                "generation-aware",
                "selected-width<=256",
                "exact-width<=64",
            }
        ),
        description="stable generation-aware bounded reuse plan",
    ),
    PACKED_GQA: KernelInputEnvelope(
        dtypes=frozenset({"bfloat16"}),
        head_dimensions=frozenset({64, 128}),
        features=frozenset(
            {"ragged-lengths", "current-stream", "caller-output-buffer"}
        ),
        description="fixed-stride ragged grouped-query attention",
    ),
}
_REFERENCE_ENVELOPES = {
    **_ENVELOPES,
    RECONSTRUCT: KernelInputEnvelope(
        dtypes=frozenset({"bfloat16"}),
        factor_ranks=frozenset({64, 128, 160, 256}),
        features=frozenset({"current-stream", "caller-output-buffer"}),
        description="selected-row pre-RoPE reconstruction for any positive head dimension",
    ),
    PACKED_GQA: KernelInputEnvelope(
        dtypes=frozenset({"bfloat16"}),
        features=frozenset(
            {"ragged-lengths", "current-stream", "caller-output-buffer"}
        ),
        description="readable fixed-stride ragged grouped-query attention",
    ),
}
_ATTRS = {
    RECONSTRUCT: "reconstruct",
    RECONSTRUCT_ROPE: "reconstruct_rope",
    PLAN_REUSE: "plan_reuse",
    PACKED_GQA: "packed_gqa",
}


def _register_candidates() -> None:
    for op, attr in _ATTRS.items():
        register_kernel(
            KernelSpec(
                op=op,
                backend=KernelBackend.TORCH,
                target=f"{_PROVIDER_ROOT}.reference:{attr}",
                operation_revision="shadowkv-v1",
                implementation_id=REFERENCE_IMPLEMENTATIONS[op],
                specialization=KernelSpecialization.REFERENCE,
                input_envelope=_REFERENCE_ENVELOPES[op],
                execution=_REFERENCE_EXECUTION,
                description="Readable Torch correctness oracle",
            )
        )
        register_kernel(
            KernelSpec(
                op=op,
                backend=KernelBackend.AOT,
                target=f"{_PROVIDER_ROOT}.aot:{attr}",
                capabilities=frozenset({Cap.CUDA}),
                operation_revision="shadowkv-v1",
                implementation_id=GENERIC_AOT_IMPLEMENTATIONS[op],
                specialization=KernelSpecialization.GENERIC,
                supported_architectures=frozenset({"sm80", "sm100a"}),
                input_envelope=_ENVELOPES[op],
                execution=(
                    _AOT_GRAPH_EXECUTION if op == PACKED_GQA else _AOT_EXECUTION
                ),
                availability_target=f"{_PROVIDER_ROOT}.aot:available",
                warmup_target=f"{_PROVIDER_ROOT}.aot:warm_up",
                qualification_references=_QUALIFICATION_REFERENCES,
                description="Generic CUDA AOT implementation qualified on SM80 and SM100a",
            )
        )


_register_candidates()


def reconstruct(
    u: Any,
    sv: Any,
    positions: Any,
    out: Any | None = None,
    *,
    implementation: str | Callable[..., Any] | None = None,
) -> Any:
    validate_reconstruct(u, sv, positions, out)
    function = resolve_callable(
        RECONSTRUCT,
        implementation,
        reference_id=REFERENCE_IMPLEMENTATIONS[RECONSTRUCT],
    )
    return function(u, sv, positions, out=out)


def reconstruct_rope(
    u: Any,
    sv: Any,
    positions: Any,
    inverse_frequencies: Any,
    out: Any | None = None,
    *,
    implementation: str | Callable[..., Any] | None = None,
) -> Any:
    validate_reconstruct_rope(u, sv, positions, inverse_frequencies, out)
    function = resolve_callable(
        RECONSTRUCT_ROPE,
        implementation,
        reference_id=REFERENCE_IMPLEMENTATIONS[RECONSTRUCT_ROPE],
    )
    return function(u, sv, positions, inverse_frequencies, out=out)


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
    implementation: str | Callable[..., Any] | None = None,
) -> ShadowKVReusePlan:
    validate_plan_reuse(
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
    function = resolve_callable(
        PLAN_REUSE,
        implementation,
        reference_id=REFERENCE_IMPLEMENTATIONS[PLAN_REUSE],
    )
    result = function(
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
        validate=validate,
    )
    if isinstance(result, ShadowKVReusePlan):
        return result
    return ShadowKVReusePlan(
        result.plan, result.deduplicated_exact_chunks, result.counts
    )


def packed_gqa(
    query: Any,
    keys: Any,
    values: Any,
    lengths: Any,
    *,
    weights: Any | None = None,
    out: Any | None = None,
    validate_lengths: bool = True,
    implementation: str | Callable[..., Any] | None = None,
) -> Any:
    validate_packed_gqa(
        query,
        keys,
        values,
        lengths,
        weights=weights,
        out=out,
        validate_lengths=validate_lengths,
    )
    function = resolve_callable(
        PACKED_GQA,
        implementation,
        reference_id=REFERENCE_IMPLEMENTATIONS[PACKED_GQA],
    )
    return function(
        query,
        keys,
        values,
        lengths,
        weights=weights,
        out=out,
        validate_lengths=validate_lengths,
    )


__all__ = [
    "GENERIC_AOT_IMPLEMENTATIONS",
    "PACKED_GQA",
    "PLAN_REUSE",
    "RECONSTRUCT",
    "RECONSTRUCT_ROPE",
    "REFERENCE_IMPLEMENTATIONS",
    "ShadowKVReusePlan",
    "bind_implementation",
    "packed_gqa",
    "plan_reuse",
    "prepare_implementation",
    "reconstruct",
    "reconstruct_rope",
]
