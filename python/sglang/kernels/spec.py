"""Lightweight metadata for the unified ``sglang.kernels`` namespace.

This module defines small, dependency-free descriptors used to *inventory*
kernel implementations and drive a simple, heuristic dispatch. It intentionally
does not import ``torch``, ``sgl_kernel`` or ``sglang.kernels.jit`` at module
import time so that ``import sglang.kernels`` stays cheap and works on a CPU-only
box (see RFC #29630, Phase 2).

The concrete callable behind a :class:`KernelSpec` is resolved lazily through
``KernelSpec.load()``; nothing is imported until a kernel is actually called.

Backend vs. device (RFC #29630 follow-up): :class:`KernelBackend` names only the
*provenance* of an implementation (how it is built / where it comes from), not
the hardware it runs on. Both JIT and AOT sources already build for CUDA *and*
ROCm, and a wheel may ship only a per-op subset, so platform support is
per-``(op, backend)`` metadata carried by :class:`CapabilityRequirement`, not
derivable from the backend name.
"""

from __future__ import annotations

import importlib
from enum import Enum
from typing import Any, Callable, ClassVar, FrozenSet, Optional, Tuple, Union

import msgspec


class KernelBackend(str, Enum):
    """Provenance of a kernel implementation (how it is built), not its device.

    ``JIT`` (``sglang.kernels.jit``, compiles under nvcc *and* hipcc) and ``AOT``
    (the ``sgl_kernel`` wheel, built for CUDA *and* ROCm) are both cross-device;
    which devices a given op supports is expressed by its
    :class:`CapabilityRequirement` list. Platform-specific libraries (e.g.
    ``aiter`` on AMD, ``torch_npu`` on Ascend) are just additional provenance
    values, each pinned to its device by its ``CapabilityRequirement``.
    """

    TORCH = "torch"  # pure-torch reference (forward_native)
    TORCH_COMPILE = "torch_compile"  # torch.compile(forward_native)
    TRITON = "triton"
    JIT = "jit"  # sglang.kernels.jit (nvcc / hipcc)
    AOT = "aot"  # sgl_kernel wheel (CUDA / ROCm builds)
    CUTE_DSL = "cute_dsl"
    FLYDSL = "flydsl"  # FlyDSL MLIR compiler (device=HIP, gfx950)
    FLASHINFER = "flashinfer"
    DEEPGEMM = "deepgemm"
    AITER = "aiter"  # AMD aiter library (device=HIP)
    TORCH_NPU = "torch_npu"  # Ascend NPU vendor runtime (device=NPU)
    # TODO(RFC #29630): more provenance as needed (cpu-avx, sgl_kernel_npu, ...)


class DeviceType(str, Enum):
    """Accelerator device family a kernel can run on."""

    CUDA = "cuda"
    HIP = "hip"
    NPU = "npu"  # Ascend NPU (torch_npu / sgl_kernel_npu)
    CPU = "cpu"
    # TODO(RFC #29630): XPU / MUSA / ... as backends land.


class KernelSpecialization(str, Enum):
    """How an implementation relates to an accelerator architecture.

    ``UNSPECIFIED`` preserves the metadata contract for registrations created
    before implementation identities were introduced. New architecture-aware
    registrations must choose ``GENERIC``, ``ARCHITECTURE``, or ``REFERENCE``.
    A generic implementation is unspecialized, but still declares every exact
    architecture on which it may be selected.
    """

    UNSPECIFIED = "unspecified"
    GENERIC = "generic"
    ARCHITECTURE = "architecture"
    REFERENCE = "reference"


def cuda_architecture_name(major: int, minor: int = 0) -> str:
    """Return the exact architecture identity used by SGLang build profiles."""
    known = {
        (8, 0): "sm80",
        (9, 0): "sm90a",
        (10, 0): "sm100a",
    }
    return known.get((major, minor), f"sm{major}{minor}")


class PlatformInfo(msgspec.Struct, frozen=True):
    """A minimal snapshot of the runtime accelerator platform.

    Kept torch-free at import time; use :meth:`detect` to build one from the
    live process (which does import ``torch``).
    """

    device_type: str = "cpu"  # "cuda", "hip", "cpu", ...
    cuda_arch_major: Optional[int] = None
    cuda_arch_minor: Optional[int] = None
    device_index: Optional[int] = None
    architecture_name: Optional[str] = None

    @property
    def device(self) -> DeviceType:
        try:
            return DeviceType(self.device_type)
        except (ValueError, TypeError):
            return DeviceType.CPU

    @property
    def is_cuda(self) -> bool:
        return self.device_type == "cuda"

    @property
    def is_hip(self) -> bool:
        return self.device_type == "hip"

    @property
    def architecture(self) -> Optional[str]:
        if self.architecture_name is not None:
            return self.architecture_name
        if self.is_cuda and self.cuda_arch_major is not None:
            return cuda_architecture_name(
                self.cuda_arch_major, self.cuda_arch_minor or 0
            )
        return None

    @classmethod
    def detect(cls, device: Optional[Any] = None) -> PlatformInfo:
        """Build a :class:`PlatformInfo` for ``device``.

        Never raises: if ``torch`` is missing or no accelerator is visible the
        default CPU platform is returned. ``device`` may be a tensor, a
        ``torch.device``, a CUDA index, or a CUDA device string. Supplying a
        tensor keeps multi-device dispatch independent from the process-wide
        current device.
        """
        try:
            import torch
        except Exception:
            return cls()

        try:
            resolved = getattr(device, "device", device)
            if isinstance(resolved, int):
                resolved = torch.device("cuda", resolved)
            elif resolved is not None:
                resolved = torch.device(resolved)

            if torch.version.hip is not None and torch.cuda.is_available():
                index = None
                if resolved is not None and resolved.type == "cuda":
                    index = resolved.index
                return cls(device_type="hip", device_index=index)
            npu = getattr(torch, "npu", None)
            if npu is not None and npu.is_available():
                return cls(device_type="npu")
            if torch.cuda.is_available():
                if resolved is not None and resolved.type != "cuda":
                    return cls(device_type=resolved.type)
                index = (
                    torch.cuda.current_device()
                    if resolved is None or resolved.index is None
                    else resolved.index
                )
                major, minor = torch.cuda.get_device_capability(index)
                return cls(
                    device_type="cuda",
                    cuda_arch_major=major,
                    cuda_arch_minor=minor,
                    device_index=index,
                    architecture_name=cuda_architecture_name(major, minor),
                )
        except Exception:
            pass
        return cls()


class CapabilityRequirement(msgspec.Struct, frozen=True):
    """One device (plus an optional CUDA-arch window) a backend can run on.

    A :class:`KernelSpec` / :class:`~sglang.kernels.fused_op.BaseFusedOp` backend
    carries a *set* of these with **OR** semantics — any matching entry makes the
    backend eligible, and an empty set means unrestricted (runs anywhere). A set
    (not a tuple) because order and duplicates are meaningless here: ``{CUDA,
    HIP}`` and ``{HIP, CUDA}`` describe the same thing. This replaces the old
    ``requires_cuda`` / ``requires_hip`` booleans (whose AND semantics could not
    express "CUDA or HIP"); arch bounds now attach to the device they describe
    (``min_cuda_arch`` / ``max_cuda_arch`` apply only when ``device == CUDA``).

    The device-only cases are so common that they are exposed as class constants
    (``CapabilityRequirement.CUDA`` / ``.HIP`` / ``.NPU``); use :meth:`cuda` for an
    arch-bounded CUDA requirement (e.g. ``CapabilityRequirement.cuda(
    min_sm=(10, 0))`` for SM100+).
    """

    device: DeviceType
    min_cuda_arch: Optional[Tuple[int, int]] = None
    max_cuda_arch: Optional[Tuple[int, int]] = None

    # Common device-only shortcuts, assigned after the class body (they are
    # instances of the class itself). ClassVar keeps them out of msgspec fields.
    CUDA: ClassVar[CapabilityRequirement]
    HIP: ClassVar[CapabilityRequirement]
    NPU: ClassVar[CapabilityRequirement]

    @classmethod
    def cuda(
        cls,
        min_sm: Optional[Tuple[int, int]] = None,
        max_sm: Optional[Tuple[int, int]] = None,
    ) -> CapabilityRequirement:
        """A CUDA requirement bounded to an SM-arch window (inclusive)."""
        return cls(device=DeviceType.CUDA, min_cuda_arch=min_sm, max_cuda_arch=max_sm)

    def is_satisfied_by(self, platform: PlatformInfo) -> bool:
        if self.device != platform.device:
            return False
        if self.device == DeviceType.CUDA and platform.cuda_arch_major is not None:
            arch = (platform.cuda_arch_major, platform.cuda_arch_minor or 0)
            if self.min_cuda_arch is not None and arch < self.min_cuda_arch:
                return False
            if self.max_cuda_arch is not None and arch > self.max_cuda_arch:
                return False
        return True


CapabilityRequirement.CUDA = CapabilityRequirement(device=DeviceType.CUDA)
CapabilityRequirement.HIP = CapabilityRequirement(device=DeviceType.HIP)
CapabilityRequirement.NPU = CapabilityRequirement(device=DeviceType.NPU)


def capabilities_satisfied(
    capabilities: Union[
        FrozenSet[CapabilityRequirement],
        Tuple[CapabilityRequirement, ...],
        CapabilityRequirement,
    ],
    platform: PlatformInfo,
) -> bool:
    """OR over ``capabilities`` (empty = unrestricted).

    Accepts a set/tuple of requirements, or tolerates a single
    :class:`CapabilityRequirement` (the pre-decouple API used one) by wrapping it.
    """
    if isinstance(capabilities, CapabilityRequirement):
        capabilities = (capabilities,)
    return (not capabilities) or any(c.is_satisfied_by(platform) for c in capabilities)


class FormatSignature(msgspec.Struct, frozen=True):
    """A light description of a kernel's data contract.

    This is deliberately loose in the first version — enough to document intent
    and support future inventory tooling, not a strict schema.
    """

    supported_dtypes: Tuple[str, ...] = ()
    in_place: bool = False
    description: str = ""


class KernelInputEnvelope(msgspec.Struct, frozen=True):
    """Serializable input limits used during initialization-time selection.

    Empty sets and ``None`` limits mean that the descriptor does not restrict
    that dimension. A request envelope normally contains one value for each
    known dimension. ``features`` carries stable operation-specific contract
    tags without forcing the shared registry to know ShadowKV tensor layouts.
    """

    dtypes: FrozenSet[str] = frozenset()
    head_dimensions: FrozenSet[int] = frozenset()
    factor_ranks: FrozenSet[int] = frozenset()
    max_batch_size: Optional[int] = None
    max_tokens: Optional[int] = None
    features: FrozenSet[str] = frozenset()
    description: str = ""

    def supports(self, request: KernelInputEnvelope) -> bool:
        def covers(allowed: FrozenSet[Any], requested: FrozenSet[Any]) -> bool:
            return not allowed or requested.issubset(allowed)

        if not covers(self.dtypes, request.dtypes):
            return False
        if not covers(self.head_dimensions, request.head_dimensions):
            return False
        if not covers(self.factor_ranks, request.factor_ranks):
            return False
        if not covers(self.features, request.features):
            return False
        if (
            self.max_batch_size is not None
            and request.max_batch_size is not None
            and request.max_batch_size > self.max_batch_size
        ):
            return False
        if (
            self.max_tokens is not None
            and request.max_tokens is not None
            and request.max_tokens > self.max_tokens
        ):
            return False
        return True


class KernelExecutionProperties(msgspec.Struct, frozen=True):
    """Execution guarantees exposed to the plan resolver."""

    deterministic: bool = False
    current_stream: bool = True
    graph_compatible: bool = False
    supports_eager: bool = True
    mutates_inputs: bool = False
    permits_output_aliasing: bool = False
    workspace_description: str = ""


class KernelSpec(msgspec.Struct, frozen=True):
    """A single callable kernel implementation and its metadata.

    Parameters
    ----------
    op:
        Fully-qualified operator id, ``"<group>.<name>"`` (e.g.
        ``"layernorm.rmsnorm"``). This is the public lookup key.
    backend:
        Which :class:`KernelBackend` (provenance) provides this implementation.
    target:
        Import path of the callable in ``"module:attr"`` form, resolved lazily
        by :meth:`load` (e.g. ``"sgl_kernel:rmsnorm"``). ``attr`` may be a
        dotted path into a module-level object, e.g.
        ``"sglang.kernels.ops.layernorm:_RMSNORM.forward_aot"`` for a bound
        :class:`~sglang.kernels.fused_op.BaseFusedOp` backend method.
    capabilities:
        Set of :class:`CapabilityRequirement` (OR semantics; empty = runs on
        any device) used by the selector to skip backends unusable on the
        detected platform.
    format_signature:
        Optional data-contract description for inventory/documentation.
    description:
        Human-readable one-liner.
    """

    op: str
    backend: KernelBackend
    target: str
    capabilities: FrozenSet[CapabilityRequirement] = frozenset()
    format_signature: FormatSignature = msgspec.field(default_factory=FormatSignature)
    description: str = ""
    operation_revision: str = "legacy-v1"
    implementation_id: Optional[str] = None
    specialization: KernelSpecialization = KernelSpecialization.UNSPECIFIED
    supported_architectures: FrozenSet[str] = frozenset()
    input_envelope: KernelInputEnvelope = msgspec.field(
        default_factory=KernelInputEnvelope
    )
    execution: KernelExecutionProperties = msgspec.field(
        default_factory=KernelExecutionProperties
    )
    availability_target: Optional[str] = None
    warmup_target: Optional[str] = None
    qualification_references: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.op or not self.target:
            raise ValueError("KernelSpec requires non-empty op and target values")
        if self.implementation_id is not None and not self.implementation_id:
            raise ValueError("KernelSpec implementation_id cannot be empty")
        invalid_architectures = {
            architecture
            for architecture in self.supported_architectures
            if not architecture or architecture.lower() in {"*", "any", "all"}
        }
        if invalid_architectures:
            raise ValueError(
                "KernelSpec supported_architectures must use exact identities: "
                f"{sorted(invalid_architectures)}"
            )
        if self.specialization in {
            KernelSpecialization.GENERIC,
            KernelSpecialization.ARCHITECTURE,
        }:
            if self.implementation_id is None:
                raise ValueError(
                    "Architecture-aware KernelSpec requires implementation_id"
                )
            if not self.supported_architectures:
                raise ValueError(
                    "Generic and architecture-specialized KernelSpec entries "
                    "require explicit supported_architectures"
                )
        if (
            self.specialization is KernelSpecialization.ARCHITECTURE
            and len(self.supported_architectures) != 1
        ):
            raise ValueError(
                "Architecture-specialized KernelSpec requires exactly one architecture"
            )

    @property
    def group(self) -> str:
        return self.op.split(".", 1)[0]

    @property
    def name(self) -> str:
        return self.op.split(".", 1)[1] if "." in self.op else self.op

    @property
    def provider(self) -> KernelBackend:
        """Provider provenance, retained under the legacy ``backend`` field."""
        return self.backend

    @property
    def identity(self) -> str:
        """Stable implementation identity, including a legacy-safe default."""
        return self.implementation_id or self.backend.value

    def rejection_reasons(
        self,
        platform: PlatformInfo,
        *,
        envelope: Optional[KernelInputEnvelope] = None,
        require_graph: bool = False,
    ) -> Tuple[str, ...]:
        """Return deterministic metadata rejection reasons for one request."""
        reasons = []
        if not capabilities_satisfied(self.capabilities, platform):
            reasons.append("device-capability")
        if (
            self.supported_architectures
            and platform.architecture not in self.supported_architectures
        ):
            reasons.append(f"architecture:{platform.architecture or 'unknown'}")
        if envelope is not None and not self.input_envelope.supports(envelope):
            reasons.append("input-envelope")
        if require_graph and not self.execution.graph_compatible:
            reasons.append("graph-incompatible")
        if not self.execution.supports_eager and not require_graph:
            reasons.append("eager-incompatible")
        return tuple(reasons)

    def is_available(
        self,
        platform: PlatformInfo,
        *,
        envelope: Optional[KernelInputEnvelope] = None,
        require_graph: bool = False,
    ) -> bool:
        """Whether this implementation can run under the declared request."""
        return not self.rejection_reasons(
            platform, envelope=envelope, require_graph=require_graph
        )

    @staticmethod
    def _load_target(target: str) -> Callable:
        module_path, sep, attr = target.partition(":")
        if not sep or not attr:
            raise ValueError(f"KernelSpec target must be 'module:attr', got {target!r}")
        obj = importlib.import_module(module_path)
        for part in attr.split("."):
            obj = getattr(obj, part)
        return obj

    def load(self) -> Callable:
        """Import and return the backing callable.

        Raises the underlying ``ImportError`` / ``AttributeError`` if the
        backend is not installed on this platform — call sites decide how to
        handle that.
        """
        return self._load_target(self.target)

    def provider_available(self) -> bool:
        """Evaluate the optional lazy provider probe.

        Inventory never calls this method, so listing candidates stays free of
        CUDA initialization and provider imports.
        """
        if self.availability_target is None:
            return True
        return bool(self._load_target(self.availability_target)())

    def warm_up(self, *args: Any, **kwargs: Any) -> None:
        """Run the optional provider preparation hook before graph capture."""
        if self.warmup_target is not None:
            self._load_target(self.warmup_target)(*args, **kwargs)

    def inventory_record(self) -> dict[str, Any]:
        """Return stable, JSON-compatible metadata without loading providers."""
        record = msgspec.json.decode(msgspec.json.encode(self))
        record["implementation_id"] = self.identity
        record["provider"] = self.provider.value
        return record
