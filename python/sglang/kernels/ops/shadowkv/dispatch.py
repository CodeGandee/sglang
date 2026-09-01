"""Lazy binding helpers for ShadowKV implementation candidates."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sglang.kernels.registry import registry
from sglang.kernels.spec import KernelSpec


def implementation_spec(op: str, implementation_id: str) -> KernelSpec:
    if not op.startswith("shadowkv."):
        raise ValueError(f"not a ShadowKV operation: {op!r}")
    return registry.get_implementation(op, implementation_id)


def bind_implementation(op: str, implementation_id: str) -> Callable[..., Any]:
    """Load and bind one exact provider implementation."""
    return implementation_spec(op, implementation_id).load()


def prepare_implementation(
    op: str,
    implementation_id: str,
    *warmup_args: Any,
    **warmup_kwargs: Any,
) -> KernelSpec:
    """Probe, load, and warm one provider before capture or measurement."""
    spec = implementation_spec(op, implementation_id)
    if not spec.provider_available():
        raise RuntimeError(
            f"provider unavailable for {op!r} implementation {implementation_id!r}"
        )
    spec.load()
    spec.warm_up(*warmup_args, **warmup_kwargs)
    return spec


def resolve_callable(
    op: str,
    implementation: str | Callable[..., Any] | None,
    *,
    reference_id: str,
) -> Callable[..., Any]:
    if implementation is None:
        return bind_implementation(op, reference_id)
    if isinstance(implementation, str):
        return bind_implementation(op, implementation)
    if not callable(implementation):
        raise TypeError("implementation must be an implementation ID or callable")
    return implementation
