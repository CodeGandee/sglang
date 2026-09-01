"""Device-aware fixed-path kernel resolution over the :data:`registry`.

There is no priority *ranking* or preference heuristic. Resolution of an op's
call path is deterministic:

- an op with a single registered backend resolves to it directly;
- an op with several registered backends is filtered by the detected platform
  (a hard :class:`~sglang.kernels.spec.CapabilityRequirement` check, not a
  preference). If exactly one backend is usable on this device, it is the fixed
  call path; if several remain usable, the caller must name the backend
  explicitly (``backend=...``).

Because ``KernelBackend`` is now device-agnostic provenance, the *same* backend
(e.g. ``AOT``) may be registered for an op on more than one device; the
availability filter is what makes ``select_kernel`` pick the right one per
platform. Filtering by device is a hard eligibility gate, not the ranked
auto-selection that ``BaseFusedOp`` performs.

:func:`get_kernel` is the fast path used by the public ``ops.*`` wrappers: it
resolves the spec to its callable and caches the result so repeated calls do
not re-run resolution or re-import (the platform is constant per process).
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Any, Callable, FrozenSet, Optional, Sequence, Tuple

import msgspec

from sglang.kernels.registry import registry
from sglang.kernels.spec import (
    KernelBackend,
    KernelInputEnvelope,
    KernelSpec,
    KernelSpecialization,
    PlatformInfo,
)


class KernelSelectionPolicy(str, Enum):
    """Initialization-time implementation policy."""

    AUTO = "auto"
    EXACT = "exact"
    GENERIC = "generic"
    REFERENCE = "reference"


class CandidateRejection(msgspec.Struct, frozen=True):
    """Why one registered candidate was excluded from a request."""

    implementation_id: str
    provider: str
    specialization: str
    reasons: Tuple[str, ...]


class KernelSelection(msgspec.Struct, frozen=True):
    """Serializable result of deterministic candidate selection."""

    op: str
    policy: str
    platform: PlatformInfo
    envelope: KernelInputEnvelope
    selected: KernelSpec
    fallback_reason: Optional[str] = None
    rejections: Tuple[CandidateRejection, ...] = ()


class KernelSelectionError(RuntimeError):
    """Raised when a fixed implementation cannot be selected safely."""

    def __init__(
        self,
        message: str,
        *,
        op: str,
        failure_class: str,
        rejections: Sequence[CandidateRejection] = (),
    ) -> None:
        super().__init__(message)
        self.op = op
        self.failure_class = failure_class
        self.rejections = tuple(rejections)


@lru_cache(maxsize=1)
def _platform() -> PlatformInfo:
    return PlatformInfo.detect()


def select_kernel(op: str, backend: Optional[KernelBackend] = None) -> KernelSpec:
    """Return the :class:`KernelSpec` for ``op`` (its fixed call path).

    Parameters
    ----------
    op:
        Operator id, ``"<group>.<name>"``.
    backend:
        Required only when ``op`` has more than one backend *usable on the
        current device*; selects which one. Otherwise optional.

    Raises
    ------
    KeyError
        If ``op`` is unknown, or if ``backend`` is requested but not registered.
    ValueError
        If ``op`` has multiple device-eligible backends and ``backend`` is not
        given, or if none are eligible on this platform.
    """
    specs = registry.get(op)
    if not specs:
        raise KeyError(f"No kernels registered for op {op!r}")

    if backend is not None:
        return registry.get_backend(op, backend)

    if len(specs) == 1:
        return specs[0]

    # Multiple backends: hard-filter by device eligibility.
    platform = _platform()
    eligible = [s for s in specs if s.is_available(platform)]
    if len(eligible) == 1:
        return eligible[0]
    if not eligible:
        raise ValueError(
            f"op {op!r} has no backend usable on device {platform.device.value!r} "
            f"(registered: {[s.backend.value for s in specs]})"
        )
    raise ValueError(
        f"op {op!r} has multiple backends usable on device "
        f"{platform.device.value!r} ({[s.backend.value for s in eligible]}); "
        f"pass backend=... to choose one"
    )


def _selection_platform(
    *, platform: Optional[PlatformInfo], device: Optional[Any]
) -> PlatformInfo:
    if platform is not None and device is not None:
        raise ValueError("Pass platform or device, not both")
    if platform is not None:
        return platform
    if device is not None:
        return PlatformInfo.detect(device)
    return _platform()


def select_kernel_candidate(
    op: str,
    *,
    policy: KernelSelectionPolicy = KernelSelectionPolicy.AUTO,
    implementation_id: Optional[str] = None,
    provider: Optional[KernelBackend] = None,
    platform: Optional[PlatformInfo] = None,
    device: Optional[Any] = None,
    envelope: Optional[KernelInputEnvelope] = None,
    require_graph: bool = False,
    packaged_implementation_ids: Optional[FrozenSet[str]] = None,
    qualified_implementation_ids: Optional[FrozenSet[str]] = None,
    preferred_implementation_ids: Sequence[str] = (),
    probe_provider: bool = False,
) -> KernelSelection:
    """Select one architecture-aware implementation deterministically.

    Selection is metadata-only unless ``probe_provider`` is true. The caller
    supplies artifact and qualification identities independently so package
    availability cannot be confused with correctness promotion.
    """
    if not isinstance(policy, KernelSelectionPolicy):
        policy = KernelSelectionPolicy(policy)
    if implementation_id is not None and policy is not KernelSelectionPolicy.EXACT:
        raise ValueError("implementation_id requires the exact selection policy")
    if policy is KernelSelectionPolicy.EXACT and implementation_id is None:
        raise ValueError("exact selection requires implementation_id")

    specs = sorted(registry.get(op), key=lambda item: item.identity)
    if not specs:
        raise KernelSelectionError(
            f"No implementations are registered for op {op!r}",
            op=op,
            failure_class="unknown-operation",
        )

    selected_platform = _selection_platform(platform=platform, device=device)
    selected_envelope = envelope or KernelInputEnvelope()
    rejections = []
    eligible = []
    preference = {
        identity: position
        for position, identity in enumerate(preferred_implementation_ids)
    }

    for spec in specs:
        reasons = list(
            spec.rejection_reasons(
                selected_platform,
                envelope=selected_envelope,
                require_graph=require_graph,
            )
        )
        if implementation_id is not None and spec.identity != implementation_id:
            reasons.append("not-requested-implementation")
        if provider is not None and spec.provider is not provider:
            reasons.append(f"provider-policy:{provider.value}")
        if (
            packaged_implementation_ids is not None
            and spec.identity not in packaged_implementation_ids
        ):
            reasons.append("not-packaged")
        if (
            qualified_implementation_ids is not None
            and spec.identity not in qualified_implementation_ids
        ):
            reasons.append("unqualified")
        if policy is KernelSelectionPolicy.GENERIC:
            if spec.specialization is not KernelSpecialization.GENERIC:
                reasons.append("generic-only-policy")
        elif policy is KernelSelectionPolicy.REFERENCE:
            if spec.specialization is not KernelSpecialization.REFERENCE:
                reasons.append("reference-only-policy")
        elif policy is KernelSelectionPolicy.AUTO:
            if spec.specialization is KernelSpecialization.REFERENCE:
                reasons.append("reference-requires-explicit-policy")

        if not reasons and probe_provider:
            try:
                available = spec.provider_available()
            except Exception as exc:
                raise KernelSelectionError(
                    f"Provider probe failed for {op!r} implementation "
                    f"{spec.identity!r}: {exc}",
                    op=op,
                    failure_class="provider-probe-failure",
                    rejections=rejections,
                ) from exc
            if not available:
                reasons.append("provider-unavailable")

        if reasons:
            rejections.append(
                CandidateRejection(
                    implementation_id=spec.identity,
                    provider=spec.provider.value,
                    specialization=spec.specialization.value,
                    reasons=tuple(sorted(set(reasons))),
                )
            )
        else:
            eligible.append(spec)

    if not eligible:
        requested = implementation_id or policy.value
        raise KernelSelectionError(
            f"No eligible implementation for op {op!r} under request {requested!r}",
            op=op,
            failure_class=(
                "explicit-request-unavailable"
                if policy is not KernelSelectionPolicy.AUTO
                else "no-qualified-candidate"
            ),
            rejections=rejections,
        )

    def tier(spec: KernelSpec) -> int:
        if policy in {
            KernelSelectionPolicy.EXACT,
            KernelSelectionPolicy.GENERIC,
            KernelSelectionPolicy.REFERENCE,
        }:
            return 0
        if spec.specialization is KernelSpecialization.ARCHITECTURE:
            return 0
        if spec.specialization is KernelSpecialization.GENERIC:
            return 1
        return 2

    best_tier = min(tier(spec) for spec in eligible)
    preferred = [spec for spec in eligible if tier(spec) == best_tier]
    ranked = sorted(
        preferred,
        key=lambda spec: (
            preference.get(spec.identity, len(preference)),
            spec.identity,
        ),
    )
    if len(ranked) > 1:
        first_rank = preference.get(ranked[0].identity, len(preference))
        second_rank = preference.get(ranked[1].identity, len(preference))
        if first_rank == second_rank:
            identities = sorted(spec.identity for spec in ranked)
            raise KernelSelectionError(
                f"Equal-preference implementations for op {op!r}: {identities}",
                op=op,
                failure_class="ambiguous-preference",
                rejections=rejections,
            )

    selected = ranked[0]
    fallback_reason = None
    if (
        policy is KernelSelectionPolicy.AUTO
        and selected.specialization is KernelSpecialization.GENERIC
    ):
        architecture = selected_platform.architecture or "unknown"
        fallback_reason = (
            f"no qualified architecture specialization selected for {architecture}"
        )
    return KernelSelection(
        op=op,
        policy=policy.value,
        platform=selected_platform,
        envelope=selected_envelope,
        selected=selected,
        fallback_reason=fallback_reason,
        rejections=tuple(rejections),
    )


@lru_cache(maxsize=None)
def _resolve(op: str, backend: Optional[KernelBackend]) -> Callable:
    return select_kernel(op, backend=backend).load()


def get_kernel(op: str, backend: Optional[KernelBackend] = None) -> Callable:
    """Resolve ``op`` to a callable kernel and cache it.

    This is what the public ``sglang.kernels.ops.*`` wrappers call. The first
    call resolves and imports the backend; later calls hit the cache.
    """
    return _resolve(op, backend)


def clear_cache() -> None:
    """Drop the resolved-callable cache (used by tests)."""
    _resolve.cache_clear()
