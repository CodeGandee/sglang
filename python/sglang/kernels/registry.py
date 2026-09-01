"""In-memory registry of :class:`KernelSpec` entries.

The registry is the single inventory of "which operators have which backend
implementations". It is populated at import time by the ``sglang.kernels.ops.*``
group packages, using only metadata (import path strings) — registering a spec
never imports ``torch`` or a kernel backend.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from sglang.kernels.spec import KernelBackend, KernelSpec


class KernelRegistry:
    """Maps ``"<group>.<name>"`` operator ids to their :class:`KernelSpec` list."""

    def __init__(self) -> None:
        self._by_op: Dict[str, List[KernelSpec]] = defaultdict(list)

    def register(self, spec: KernelSpec) -> KernelSpec:
        """Register ``spec``.

        Re-registering an identical spec is idempotent so that module reloads
        during tests remain safe. A different spec for the same ``(op,
        implementation_id)`` pair is rejected because silently replacing it
        makes selection depend on import order. Legacy specs derive their
        implementation identity from the backend, which preserves the former
        one-implementation-per-backend behavior.
        """
        existing = self._by_op[spec.op]
        for other in existing:
            if other.identity == spec.identity:
                if other != spec:
                    raise ValueError(
                        f"Conflicting kernel registration for op {spec.op!r}, "
                        f"implementation {spec.identity!r}: "
                        f"{other.target!r} != {spec.target!r}"
                    )
                return spec
        existing.append(spec)
        return spec

    def get(self, op: str) -> List[KernelSpec]:
        """All registered specs for ``op`` (empty list if none)."""
        return list(self._by_op.get(op, ()))

    def get_backend(self, op: str, backend: KernelBackend) -> KernelSpec:
        """The spec for ``op`` provided by ``backend``.

        Raises ``KeyError`` if no such implementation is registered and
        ``ValueError`` when several implementation identities share the
        provider. Call :meth:`get_implementation` in the ambiguous case.
        """
        matches = [spec for spec in self._by_op.get(op, ()) if spec.backend == backend]
        if not matches:
            raise KeyError(f"No '{backend.value}' backend registered for op {op!r}")
        if len(matches) != 1:
            identities = sorted(spec.identity for spec in matches)
            raise ValueError(
                f"Backend {backend.value!r} is ambiguous for op {op!r}: "
                f"implementations {identities}; select an implementation_id"
            )
        return matches[0]

    def get_implementation(self, op: str, implementation_id: str) -> KernelSpec:
        """Return the uniquely identified implementation for ``op``."""
        for spec in self._by_op.get(op, ()):
            if spec.identity == implementation_id:
                return spec
        raise KeyError(
            f"No implementation {implementation_id!r} registered for op {op!r}"
        )

    def has(self, op: str) -> bool:
        return bool(self._by_op.get(op))

    def ops(self) -> List[str]:
        """Sorted list of all registered operator ids."""
        return sorted(self._by_op.keys())

    def all_specs(self) -> List[KernelSpec]:
        specs: List[KernelSpec] = []
        for op in self.ops():
            specs.extend(self._by_op[op])
        return specs

    def inventory(self, op: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return deterministic metadata records without loading providers."""
        specs = self.get(op) if op is not None else self.all_specs()
        return [
            spec.inventory_record()
            for spec in sorted(specs, key=lambda item: (item.op, item.identity))
        ]


# Process-wide registry. Group packages register into this instance on import.
registry = KernelRegistry()


def register_kernel(spec: KernelSpec) -> KernelSpec:
    """Register ``spec`` in the process-wide :data:`registry`."""
    return registry.register(spec)
