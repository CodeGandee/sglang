"""Aux hidden states captured for Eagle3/DFlash draft models."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, List, Optional, Union

import torch

# Two representations coexist: models migrated to AuxHiddenStatePacker pass one
# packed [tokens, K * hidden] tensor, the rest still pass a list of K tensors.
AuxHiddenStates = Union[torch.Tensor, List[torch.Tensor]]

_aux_append_observer: Optional[Callable[["AuxHiddenStatePacker"], None]] = None


def set_aux_append_observer(
    observer: Optional[Callable[["AuxHiddenStatePacker"], None]],
) -> None:
    """Install an explicit post-copy observer for one active DFlash target binding."""
    global _aux_append_observer
    _aux_append_observer = observer


_aux_replay_observer: Optional[Callable[["AuxHiddenStatePacker"], None]] = None
_aux_capture: ContextVar[Optional["AuxPackerCapture"]] = ContextVar(
    "native_aux_graph_capture", default=None
)


def set_aux_replay_observer(
    observer: Optional[Callable[["AuxHiddenStatePacker"], None]],
) -> None:
    """Install a replay notification for the active target binding.

    Parameters
    ----------
    observer : callable or None
        Receives an actual completed captured packer after successful replay.
    """
    global _aux_replay_observer
    _aux_replay_observer = observer


class AuxPackerCapture:
    """Retain completed packers created during one actual graph capture.

    Notes
    -----
    The backend owns this scope only around capture, excluding warmups. Retained
    buffers live with that graph shape and are published after its replay.
    """

    def __init__(self) -> None:
        self._packers: List[AuxHiddenStatePacker] = []

    @property
    def packers(self) -> tuple["AuxHiddenStatePacker", ...]:
        """Return the actual completed native packers in capture order."""
        return tuple(self._packers)

    def record(self, packer: "AuxHiddenStatePacker") -> None:
        """Retain an actual packer once after its final original append.

        Parameters
        ----------
        packer : AuxHiddenStatePacker
            Owner whose copied buffer and complete count remain unchanged.
        """
        if any(value is packer for value in self._packers):
            raise RuntimeError("duplicate completed native auxiliary packer")
        packer.finalize()
        self._packers.append(packer)


@contextmanager
def capture_aux_packers() -> Iterator[AuxPackerCapture]:
    """Scope native buffer retention to actual capture and clear it on failure.

    Yields
    ------
    AuxPackerCapture
        Native owners retained by the backend after successful capture.
    """
    if _aux_capture.get() is not None:
        raise RuntimeError("nested native auxiliary graph capture")
    retained = AuxPackerCapture()
    token = _aux_capture.set(retained)
    try:
        yield retained
    except BaseException:
        retained._packers.clear()
        raise
    finally:
        _aux_capture.reset(token)


def publish_aux_replay(packers: Sequence["AuxHiddenStatePacker"]) -> None:
    """Publish completed stable packer buffers after successful graph replay.

    Parameters
    ----------
    packers : sequence of AuxHiddenStatePacker
        Exact retained owners for the shape that has just replayed. No append,
        copy, cursor mutation or synchronization is introduced here.
    """
    if len({id(packer) for packer in packers}) != len(packers):
        raise RuntimeError("duplicate native auxiliary graph replay owner")
    for packer in packers:
        packer.finalize()
    if _aux_replay_observer is not None:
        for packer in packers:
            _aux_replay_observer(packer)


class AuxHiddenStatePacker:
    """Drop-in for the ``[]`` a model collects Eagle3/DFlash captures into.

    Each ``.append()`` writes into one preallocated ``[tokens, K * hidden]``
    buffer, avoiding the list path's transient ~2x HBM at ``torch.cat``.
    Assumes all captures share leading shape and feature size.
    """

    # ``append`` copies, so producers need not clone a tensor they later mutate.
    copies_on_append = True

    def __init__(self, num_captures: int) -> None:
        self._num_captures = int(num_captures)
        self._buffer: Optional[torch.Tensor] = None
        self._feature_size: Optional[int] = None
        self._idx = 0

    def append(self, hidden: torch.Tensor) -> None:
        feature_size = int(hidden.shape[-1])
        if self._buffer is None:
            self._feature_size = feature_size
            self._buffer = hidden.new_empty(
                (*hidden.shape[:-1], feature_size * self._num_captures)
            )
        start = self._idx * self._feature_size
        self._buffer[..., start : start + self._feature_size].copy_(hidden)
        self._idx += 1
        if _aux_append_observer is not None:
            _aux_append_observer(self)
        captured = _aux_capture.get()
        if captured is not None and self._idx == self._num_captures:
            captured.record(self)

    def __len__(self) -> int:
        return self._idx

    def finalize(self) -> torch.Tensor:
        """Return the packed buffer; callers guard the empty case on ``len()``."""
        assert (
            self._buffer is not None and self._idx == self._num_captures
        ), f"captured {self._idx} of {self._num_captures} aux hidden states"
        return self._buffer


# What a model hands down the capture path: a plain list, or a packer writing in place.
AuxHiddenStateAccumulator = Union[List[torch.Tensor], AuxHiddenStatePacker]


def pack_aux_hidden_states(aux_hidden_states: AuxHiddenStates) -> torch.Tensor:
    if isinstance(aux_hidden_states, torch.Tensor):
        return aux_hidden_states
    return torch.cat(aux_hidden_states, dim=-1)
