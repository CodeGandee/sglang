"""Adapter from stable ShadowKV contracts to the optional native wheel."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from sglang.kernels.ops.shadowkv.contracts import ShadowKVReusePlan


def available() -> bool:
    """Return whether all generic native ShadowKV operators are installed."""
    try:
        kernels = import_module("sgl_kernel")
    except ImportError:
        return False
    probe = getattr(kernels, "shadowkv_kernels_available", None)
    return bool(callable(probe) and probe())


def warm_up(*_args: Any, **_kwargs: Any) -> None:
    """Load and validate the native extension before capture or measurement."""
    if not available():
        raise RuntimeError(
            "the installed sglang-kernel wheel has no generic ShadowKV extension"
        )


def reconstruct(*args: Any, **kwargs: Any) -> Any:
    return import_module("sgl_kernel").shadowkv_reconstruct(*args, **kwargs)


def reconstruct_rope(*args: Any, **kwargs: Any) -> Any:
    return import_module("sgl_kernel").shadowkv_reconstruct_rope(*args, **kwargs)


def packed_gqa(*args: Any, **kwargs: Any) -> Any:
    return import_module("sgl_kernel").shadowkv_packed_gqa(*args, **kwargs)


def plan_reuse(*args: Any, **kwargs: Any) -> ShadowKVReusePlan:
    result = import_module("sgl_kernel").shadowkv_plan_reuse(*args, **kwargs)
    return ShadowKVReusePlan(
        plan=result.plan,
        deduplicated_exact_chunks=result.deduplicated_exact_chunks,
        counts=result.counts,
    )
