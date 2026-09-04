"""Transactional construction of full-attention children for composite backends."""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.model_executor.model_runner import ModelRunner


CHILD_BACKEND_FACTORY_SCHEMA_VERSION = 1
_CONSTRUCTION_CONTEXT_ATTRIBUTE = "_sglang_child_backend_construction_context"
_TEMPORARY_RUNNER_ATTRIBUTES = (
    "init_new_workspace",
    "prefill_attention_backend_str",
    "decode_attention_backend_str",
    _CONSTRUCTION_CONTEXT_ATTRIBUTE,
)
_MISSING = object()


class AttentionPhaseRole(str, enum.Enum):
    """A phase that a composite backend delegates to one full-attention child."""

    PREFILL = "prefill"
    DECODE = "decode"
    TARGET_VERIFY = "target-verify"
    DRAFT_EXTEND = "draft-extend"
    EXACT_FALLBACK = "exact-fallback"


class AttentionWorkspacePolicy(str, enum.Enum):
    """Whether the provider reuses the runner workspace or creates its own."""

    SHARED = "shared"
    DEDICATED = "dedicated"


@dataclass(frozen=True, slots=True)
class FullAttentionCacheAccessContract:
    """The exact runner cache objects a child is authorized to access."""

    schema_version: int
    contract_id: str
    request_pool: object
    token_pool: object
    allocator: object
    page_size: int
    kv_cache_dtype: object

    def validate_for_runner(self, model_runner: ModelRunner) -> None:
        if self.schema_version != CHILD_BACKEND_FACTORY_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported child-backend cache contract schema: "
                f"{self.schema_version}"
            )
        if not self.contract_id.strip():
            raise ValueError("Child-backend cache contract ID must not be empty")
        expected_objects = (
            ("request_pool", self.request_pool, "req_to_token_pool"),
            ("token_pool", self.token_pool, "token_to_kv_pool"),
            ("allocator", self.allocator, "token_to_kv_pool_allocator"),
        )
        for contract_name, expected, runner_name in expected_objects:
            if getattr(model_runner, runner_name, _MISSING) is not expected:
                raise ValueError(
                    f"Child-backend {contract_name} does not match "
                    f"model_runner.{runner_name}"
                )
        if int(model_runner.page_size) != self.page_size:
            raise ValueError("Child-backend page size does not match the model runner")
        if model_runner.kv_cache_dtype != self.kv_cache_dtype:
            raise ValueError(
                "Child-backend KV-cache dtype does not match the model runner"
            )


@dataclass(frozen=True, slots=True)
class FullAttentionChildBackendRequest:
    """Frozen construction inputs for one named full-attention child."""

    schema_version: int
    backend_name: str
    effective_backend_identity: str
    phase_roles: tuple[AttentionPhaseRole, ...]
    workspace_policy: AttentionWorkspacePolicy
    cache_access: FullAttentionCacheAccessContract

    def __post_init__(self) -> None:
        if self.schema_version != CHILD_BACKEND_FACTORY_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported child-backend request schema: {self.schema_version}"
            )
        if not self.backend_name.strip():
            raise ValueError("Child-backend name must not be empty")
        if not self.effective_backend_identity.strip():
            raise ValueError("Effective child-backend identity must not be empty")
        if not self.phase_roles:
            raise ValueError("A child backend must own at least one phase role")
        if len(self.phase_roles) != len(set(self.phase_roles)):
            raise ValueError("Child-backend phase roles must be unique")


@dataclass(slots=True)
class ChildBackendConstructionContext:
    """Provider-visible transaction with explicit rollback and cleanup hooks."""

    request: FullAttentionChildBackendRequest
    _rollback_callbacks: list[Callable[[], None]] = field(default_factory=list)
    _cleanup_callbacks: list[Callable[[], None]] = field(default_factory=list)

    def register_rollback(self, callback: Callable[[], None]) -> None:
        self._rollback_callbacks.append(callback)

    def register_cleanup(self, callback: Callable[[], None]) -> None:
        self._cleanup_callbacks.append(callback)

    def rollback(self) -> None:
        _run_callbacks(self._rollback_callbacks)
        self._rollback_callbacks.clear()
        self._cleanup_callbacks.clear()

    def cleanup(self) -> None:
        _run_callbacks(self._cleanup_callbacks)
        self._cleanup_callbacks.clear()
        self._rollback_callbacks.clear()


@dataclass(slots=True)
class FullAttentionChildBackend:
    """A constructed child plus the resources owned by its transaction."""

    backend: AttentionBackend
    request: FullAttentionChildBackendRequest
    _context: ChildBackendConstructionContext = field(repr=False)
    _runner: ModelRunner = field(repr=False)
    _runner_mutations: dict[str, object] = field(repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def effective_backend_identity(self) -> str:
        return self.request.effective_backend_identity

    @property
    def phase_roles(self) -> tuple[AttentionPhaseRole, ...]:
        return self.request.phase_roles

    def close(self) -> None:
        """Release provider-owned resources once and restore runner mutations."""

        if self._closed:
            return
        self._closed = True
        first_error: Exception | None = None
        provider_close = getattr(self.backend, "close", None)
        if callable(provider_close):
            try:
                provider_close()
            except Exception as error:  # noqa: BLE001 - cleanup must continue
                first_error = error
        try:
            self._context.cleanup()
        except Exception as error:  # noqa: BLE001 - cleanup must continue
            if first_error is None:
                first_error = error
        _restore_runner_mutations(self._runner, self._runner_mutations)
        if first_error is not None:
            raise first_error


def get_child_backend_construction_context(
    model_runner: ModelRunner,
) -> ChildBackendConstructionContext | None:
    """Return the active provider-construction transaction, when present."""

    context = getattr(model_runner, _CONSTRUCTION_CONTEXT_ATTRIBUTE, None)
    return context if isinstance(context, ChildBackendConstructionContext) else None


def construct_full_attention_child_backend(
    *,
    model_runner: ModelRunner,
    request: FullAttentionChildBackendRequest,
) -> FullAttentionChildBackend:
    """Construct one registry backend without changing the outer backend identity."""

    request.cache_access.validate_for_runner(model_runner)
    from sglang.srt.layers.attention.attention_registry import (
        resolve_attention_backend_factory,
    )

    factory = resolve_attention_backend_factory(request.backend_name)
    before = dict(vars(model_runner))
    context = ChildBackendConstructionContext(request=request)
    try:
        model_runner.init_new_workspace = (
            request.workspace_policy is AttentionWorkspacePolicy.DEDICATED
        )
        model_runner.prefill_attention_backend_str = request.backend_name
        model_runner.decode_attention_backend_str = request.backend_name
        setattr(model_runner, _CONSTRUCTION_CONTEXT_ATTRIBUTE, context)
        backend = factory(model_runner)
        if backend is None:
            raise TypeError(
                f"Attention backend {request.backend_name!r} returned no backend"
            )
    except BaseException:
        try:
            context.rollback()
        finally:
            _restore_runner_state(model_runner, before)
        raise

    runner_mutations = _persistent_runner_mutations(model_runner, before)
    _restore_temporary_runner_attributes(model_runner, before)
    backend.child_backend_effective_identity = (  # type: ignore[attr-defined]
        request.effective_backend_identity
    )
    backend.child_backend_phase_roles = tuple(  # type: ignore[attr-defined]
        role.value for role in request.phase_roles
    )
    backend.child_backend_workspace_policy = (  # type: ignore[attr-defined]
        request.workspace_policy.value
    )
    backend.child_backend_cache_contract_id = (  # type: ignore[attr-defined]
        request.cache_access.contract_id
    )
    backend.prefill_attention_backend_str = (
        request.backend_name
        if AttentionPhaseRole.PREFILL in request.phase_roles
        else None
    )
    backend.decode_attention_backend_str = (
        request.backend_name
        if any(
            role
            in {
                AttentionPhaseRole.DECODE,
                AttentionPhaseRole.TARGET_VERIFY,
                AttentionPhaseRole.DRAFT_EXTEND,
                AttentionPhaseRole.EXACT_FALLBACK,
            }
            for role in request.phase_roles
        )
        else None
    )
    return FullAttentionChildBackend(
        backend=backend,
        request=request,
        _context=context,
        _runner=model_runner,
        _runner_mutations=runner_mutations,
    )


def _run_callbacks(callbacks: list[Callable[[], None]]) -> None:
    first_error: Exception | None = None
    for callback in reversed(callbacks):
        try:
            callback()
        except Exception as error:  # noqa: BLE001 - run every cleanup callback
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error


def _persistent_runner_mutations(
    model_runner: ModelRunner, before: dict[str, object]
) -> dict[str, object]:
    mutations: dict[str, object] = {}
    after = vars(model_runner)
    for name in set(before) | set(after):
        if name in _TEMPORARY_RUNNER_ATTRIBUTES:
            continue
        prior = before.get(name, _MISSING)
        current = after.get(name, _MISSING)
        if prior is not current:
            mutations[name] = prior
    return mutations


def _restore_temporary_runner_attributes(
    model_runner: ModelRunner, before: dict[str, object]
) -> None:
    for name in _TEMPORARY_RUNNER_ATTRIBUTES:
        _restore_attribute(model_runner, name, before.get(name, _MISSING))


def _restore_runner_mutations(
    model_runner: ModelRunner, mutations: dict[str, object]
) -> None:
    for name, prior in mutations.items():
        _restore_attribute(model_runner, name, prior)
    mutations.clear()


def _restore_runner_state(model_runner: ModelRunner, before: dict[str, object]) -> None:
    current = vars(model_runner)
    for name in tuple(current):
        if name not in before:
            delattr(model_runner, name)
    for name, value in before.items():
        setattr(model_runner, name, value)


def _restore_attribute(model_runner: ModelRunner, name: str, prior: object) -> None:
    if prior is _MISSING:
        if hasattr(model_runner, name):
            delattr(model_runner, name)
    else:
        setattr(model_runner, name, prior)
