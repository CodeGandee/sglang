"""Packaging and warm-up seam for future ShadowKV Triton providers.

No production Triton implementation is registered by this refactor. Tests and
future candidates may expose the stable operation signature from this package
and use ``warm_up`` before CUDA Graph capture.
"""

from __future__ import annotations

from typing import Any, Callable


def available() -> bool:
    try:
        __import__("triton")
    except ImportError:
        return False
    return True


def warm_up(callable_: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Compile and execute one candidate with its declared warm-up fixture."""
    if not available():
        raise RuntimeError("Triton is unavailable")
    callable_(*args, **kwargs)
