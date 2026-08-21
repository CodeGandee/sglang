# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic request-cache lifecycle events for optional cache providers."""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Protocol


class CacheBatchMutationKind(str, Enum):
    """Scheduler operations that change the row-to-request mapping."""

    FILTER = "filter"
    MERGE = "merge"


class CacheTerminalReason(str, Enum):
    """Why one request allocation stopped owning cache state."""

    COMPLETION = "completion"
    ABORT = "abort"
    RETRACTION = "retraction"
    FAILURE = "failure"


@dataclasses.dataclass(frozen=True, slots=True)
class CacheAllocation:
    """Stable identity for one use of a request-pool slot."""

    request_id: str
    request_pool_index: int
    generation: int

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("cache allocation request_id must not be empty")
        if self.request_pool_index <= 0:
            raise ValueError("cache allocation request_pool_index must be positive")
        if self.generation <= 0:
            raise ValueError("cache allocation generation must be positive")


@dataclasses.dataclass(frozen=True, slots=True)
class CacheBatchMutation:
    """A row-ordered snapshot after a scheduler batch mutation."""

    kind: CacheBatchMutationKind
    allocations: tuple[CacheAllocation, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class CacheTerminalEvent:
    """The first terminal notification for one allocation generation."""

    allocation: CacheAllocation
    reason: CacheTerminalReason
    detail: str | None = None


class CacheLifecycleCallbacks(Protocol):
    """Callbacks implemented by an optional attention cache provider."""

    def on_request_admitted(self, allocation: CacheAllocation) -> None: ...

    def on_request_published(self, allocation: CacheAllocation) -> None: ...

    def on_batch_mutated(self, mutation: CacheBatchMutation) -> None: ...

    def on_request_terminal(self, event: CacheTerminalEvent) -> None: ...


@dataclasses.dataclass(slots=True)
class _AllocationState:
    allocation: CacheAllocation
    published: bool = False
    terminal_reason: CacheTerminalReason | None = None


class CacheLifecycleDispatcher:
    """Validate callback order and deduplicate terminal events by slot generation.

    The dispatcher keeps only the latest generation for each request-pool slot.
    Its bookkeeping is therefore bounded by the request-pool capacity.
    """

    def __init__(self, callbacks: CacheLifecycleCallbacks):
        self.callbacks = callbacks
        self._slot_states: dict[int, _AllocationState] = {}

    def admit(self, allocations: tuple[CacheAllocation, ...]) -> None:
        """Publish newly allocated identities to the provider as one transaction."""

        admitted: list[CacheAllocation] = []
        try:
            for allocation in allocations:
                previous = self._slot_states.get(allocation.request_pool_index)
                if previous is not None:
                    if allocation.generation <= previous.allocation.generation:
                        raise RuntimeError(
                            "cache allocation generation must increase on slot reuse"
                        )
                    if previous.terminal_reason is None:
                        raise RuntimeError(
                            "cache request-pool slot was reused before termination"
                        )
                self._slot_states[allocation.request_pool_index] = _AllocationState(
                    allocation=allocation
                )
                admitted.append(allocation)
                self.callbacks.on_request_admitted(allocation)
        except Exception as error:
            self._terminate_after_callback_failure(admitted, error)
            raise

    def publish(self, allocations: tuple[CacheAllocation, ...]) -> None:
        """Notify the provider after scheduler-visible request state is complete."""

        published: list[CacheAllocation] = []
        try:
            for allocation in allocations:
                state = self._active_state(allocation)
                if state.published:
                    continue
                state.published = True
                published.append(allocation)
                self.callbacks.on_request_published(allocation)
        except Exception as error:
            self._terminate_after_callback_failure(published, error)
            raise

    def mutate_batch(
        self,
        kind: CacheBatchMutationKind,
        allocations: tuple[CacheAllocation, ...],
    ) -> None:
        """Publish the current row order after a real filter or merge."""

        for allocation in allocations:
            state = self._active_state(allocation)
            if not state.published:
                raise RuntimeError(
                    "cache allocation must be published before batch mutation"
                )
        self.callbacks.on_batch_mutated(
            CacheBatchMutation(kind=kind, allocations=allocations)
        )

    def terminate(
        self,
        allocation: CacheAllocation,
        reason: CacheTerminalReason,
        *,
        detail: str | None = None,
    ) -> bool:
        """Deliver at most one terminal callback for an allocation generation."""

        state = self._slot_states.get(allocation.request_pool_index)
        if state is None:
            return False
        if state.allocation.generation > allocation.generation:
            return False
        if state.allocation != allocation:
            raise RuntimeError("cache terminal identity does not match slot ownership")
        if state.terminal_reason is not None:
            return False
        state.terminal_reason = reason
        self.callbacks.on_request_terminal(
            CacheTerminalEvent(allocation=allocation, reason=reason, detail=detail)
        )
        return True

    def latest_state(
        self, request_pool_index: int
    ) -> tuple[CacheAllocation, bool, CacheTerminalReason | None] | None:
        """Return an immutable diagnostic snapshot for one slot."""

        state = self._slot_states.get(request_pool_index)
        if state is None:
            return None
        return state.allocation, state.published, state.terminal_reason

    def reset(self) -> None:
        """Reset bounded generation state when an idle request pool is cleared."""

        active = [
            state.allocation
            for state in self._slot_states.values()
            if state.terminal_reason is None
        ]
        if active:
            raise RuntimeError("cannot reset cache lifecycle with active allocations")
        self._slot_states.clear()

    def _active_state(self, allocation: CacheAllocation) -> _AllocationState:
        state = self._slot_states.get(allocation.request_pool_index)
        if state is None or state.allocation != allocation:
            raise RuntimeError("cache lifecycle event does not match slot ownership")
        if state.terminal_reason is not None:
            raise RuntimeError("cache lifecycle event followed terminal notification")
        return state

    def _terminate_after_callback_failure(
        self,
        allocations: list[CacheAllocation],
        error: Exception,
    ) -> None:
        detail = f"{type(error).__name__}: {error}"
        cleanup_errors: list[Exception] = []
        for allocation in reversed(allocations):
            try:
                self.terminate(
                    allocation,
                    CacheTerminalReason.FAILURE,
                    detail=detail,
                )
            except Exception as cleanup_error:  # noqa: BLE001
                cleanup_errors.append(cleanup_error)
        for cleanup_error in cleanup_errors:
            error.add_note(
                "cache lifecycle failure cleanup also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )


def infer_cache_terminal_reason(request: object) -> CacheTerminalReason:
    """Infer completion or abort for legacy release sites with finished requests."""

    finished_reason = getattr(request, "finished_reason", None)
    if finished_reason is not None:
        to_json = getattr(finished_reason, "to_json", None)
        if callable(to_json):
            payload = to_json()
            if isinstance(payload, dict) and payload.get("type") == "abort":
                return CacheTerminalReason.ABORT
        return CacheTerminalReason.COMPLETION
    if getattr(request, "is_retracted", False):
        return CacheTerminalReason.RETRACTION
    return CacheTerminalReason.FAILURE
