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
"""Runtime state for the breakable CUDA graph runner."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Iterator

from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.runner_backend_utils import (
    PREFILL_CUDA_GRAPH_CAPTURE_FAILED_MSG,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_in_breakable_cuda_graph = False
_forward_batch: ContextVar[ForwardBatch | None] = ContextVar(
    "breakable_graph_forward_batch", default=None
)


def get_graph_forward_batch() -> ForwardBatch | None:
    """Return the current real frame while retaining captured tensor geometry.

    Returns
    -------
    ForwardBatch, optional
        Actual native frame supplied by the current target/replay owner.
    """
    return _forward_batch.get()


@contextmanager
def graph_forward_batch_scope(forward_batch: ForwardBatch) -> Iterator[None]:
    """Delegate one actual native frame and restore it after success or failure.

    Parameters
    ----------
    forward_batch : ForwardBatch
        Original live frame. An enclosing authenticated target scope must
        delegate this same object; captured frames are never mutated.

    Yields
    ------
    None
        Scope in which eager graph breaks can read current live geometry.
    """
    current = _forward_batch.get()
    if current is not None and current is not forward_batch:
        raise RuntimeError("graph replay supplied a foreign target forward frame")
    token = _forward_batch.set(forward_batch)
    try:
        yield
    finally:
        _forward_batch.reset(token)


def delegated_replay_frame(
    live_frame: ForwardBatch | None, fallback_frame: ForwardBatch
) -> ForwardBatch:
    """Resolve the frame a nested graph replay must delegate.

    An enclosing authenticated target scope (e.g., the DFlash target hook)
    publishes the live serving frame; a nested replay must delegate that same
    object so eager break regions keep reading live geometry. When no scope
    is active, or the published frame is not the batch being executed, the
    replay keeps its own static frame and the foreign-frame refusal in
    graph_forward_batch_scope stays in force.

    Parameters
    ----------
    live_frame : ForwardBatch, optional
        The batch the nested replay is executing on behalf of.
    fallback_frame : ForwardBatch
        Static frame captured for this replay when delegation is not valid.

    Returns
    -------
    ForwardBatch
        Published live frame when it authenticates `live_frame`, otherwise
        `fallback_frame`.
    """
    published = get_graph_forward_batch()
    if published is not None and published is live_frame:
        return published
    return fallback_frame


def is_in_breakable_cuda_graph() -> bool:
    return _in_breakable_cuda_graph


@contextmanager
def enable_breakable_cuda_graph():
    """Mark the enclosed scope as inside a BCG capture/replay. Any exception
    raised inside is logged with the BCG-specific failure hint, then re-raised
    for the caller to handle."""
    global _in_breakable_cuda_graph
    _in_breakable_cuda_graph = True
    try:
        yield
    except Exception as exc:
        msg = PREFILL_CUDA_GRAPH_CAPTURE_FAILED_MSG.format(
            backend=Backend.BREAKABLE, suggestions=BCG_FAILURE_HINT
        )
        logger.error(f"{type(exc).__name__}: {exc}\n{msg}")
        raise
    finally:
        _in_breakable_cuda_graph = False


BCG_FAILURE_HINT = (
    "1. change to tc_piecewise by --cuda-graph-backend-prefill=tc_piecewise\n"
    "2. disable the prefill CUDA graph by --cuda-graph-backend-prefill=disabled\n"
    "3. if it is an OOM problem, set --mem-fraction-static to a smaller value "
    "(e.g., 0.8 or 0.7) or set --cuda-graph-max-bs-prefill to a smaller value "
    "(e.g., 2048)\n"
)
