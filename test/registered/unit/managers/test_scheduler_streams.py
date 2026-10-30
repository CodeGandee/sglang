"""Exercise the native scheduling-stream readiness boundary without CUDA."""

from __future__ import annotations

import ast
import inspect
import textwrap
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import torch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import scheduler as native

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Stream:
    def __init__(self, handle: int) -> None:
        self.cuda_stream = handle
        self.synchronize = Mock()


class _DeviceModule:
    def __init__(self, handles: tuple[int, ...], events: list[str]) -> None:
        self._handles = iter(handles)
        self._events = events
        self.created: list[_Stream] = []

    def Stream(self, *, priority: int) -> _Stream:
        assert priority == 0
        result = _Stream(next(self._handles))
        self.created.append(result)
        self._events.append(f"allocate:{result.cuda_stream}")
        return result

    @contextmanager
    def StreamContext(self, stream: _Stream) -> Iterator[None]:
        self._events.append(f"enter:{stream.cuda_stream}")
        try:
            yield
        finally:
            self._events.append(f"exit:{stream.cuda_stream}")


def _constructor_stream_boundary(scheduler: Any) -> None:
    """Execute the actual constructor statements around overlap initialization."""
    source, first_line = inspect.getsourcelines(native.Scheduler.__init__)
    constructor = ast.parse(textwrap.dedent("".join(source)))
    ast.increment_lineno(constructor, first_line - 1)
    function = cast(ast.FunctionDef, constructor.body[0])

    def calls(statement: ast.stmt, name: str) -> bool:
        return (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and statement.value.func.attr == name
        )

    begin = next(
        index
        for index, statement in enumerate(function.body)
        if calls(statement, "init_overlap")
    )
    end = next(
        index
        for index, statement in enumerate(function.body)
        if calls(statement, "maybe_init_ngram_embedding")
    )
    function.name = "actual_constructor_boundary"
    function.args = ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg="self")],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    )
    function.body = function.body[begin : end + 1]
    function.decorator_list = []
    function.returns = None
    namespace: dict[str, Any] = {}
    exec(  # noqa: S102 - execute only the inspected native constructor boundary.
        compile(
            ast.fix_missing_locations(constructor), inspect.getfile(native), "exec"
        ),
        namespace,
    )
    namespace[function.name](scheduler)


class TestSchedulerStreams(unittest.TestCase):
    """Validate actual native methods before the worker can advertise readiness."""

    def _scheduler(
        self, *, device: str = "cuda", handles: tuple[int, ...] = (17,)
    ) -> tuple[Any, _DeviceModule, list[str]]:
        scheduler = native.Scheduler.__new__(native.Scheduler)
        events: list[str] = []
        module = _DeviceModule(handles, events)
        scheduler.device = device
        scheduler.device_module = module
        scheduler.forward_stream = _Stream(11)
        scheduler.max_total_num_tokens = 1024
        scheduler.max_req_input_len = 128
        scheduler.startup_time = {}
        return scheduler, module, events

    def _run_event_loop(self, scheduler: Any, events: list[str], *, mlx: bool) -> None:
        watch = SimpleNamespace(
            install=lambda: events.append("watch-install"),
            mark_serving_started=lambda: events.append("watch-serving"),
        )

        def dispatch(actual: Any) -> None:
            self.assertIs(actual, scheduler)
            events.append("dispatch")

        with (
            patch.object(native, "use_mlx", return_value=mlx),
            patch.object(native, "is_cuda", return_value=not mlx),
            patch.object(native, "_is_hip", False),
            patch.object(native, "triton_load_watch", watch),
            patch.object(native, "dispatch_event_loop", side_effect=dispatch),
        ):
            scheduler.run_event_loop()

    def test_constructor_initializes_final_stream_before_ready(self) -> None:
        scheduler, module, events = self._scheduler(handles=(11, 17))
        scheduler.init_overlap = lambda: events.append("overlap")

        def continue_constructor() -> None:
            self.assertIs(scheduler.schedule_stream, module.created[-1])
            self.assertEqual(scheduler.schedule_stream.cuda_stream, 17)
            events.append("constructor-continued")

        scheduler.maybe_init_ngram_embedding = continue_constructor
        with (
            patch.object(native, "use_mlx", return_value=False),
            patch.object(native, "is_cuda", return_value=True),
            patch.object(native, "_is_hip", False),
        ):
            _constructor_stream_boundary(scheduler)
        prepared = scheduler.schedule_stream
        self.assertEqual(scheduler.get_init_info()["status"], "ready")
        events.append("ready")
        self._run_event_loop(scheduler, events, mlx=False)
        self.assertIs(scheduler.schedule_stream, prepared)
        self.assertEqual(len(module.created), 2)
        self.assertEqual(
            events,
            [
                "overlap",
                "allocate:11",
                "allocate:17",
                "constructor-continued",
                "ready",
                "watch-install",
                "watch-serving",
                "enter:17",
                "dispatch",
                "exit:17",
            ],
        )

    def test_event_loop_preserves_already_qualified_stream(self) -> None:
        scheduler, module, events = self._scheduler()
        prepared = _Stream(23)
        scheduler.schedule_stream = prepared
        self._run_event_loop(scheduler, events, mlx=False)
        self.assertIs(scheduler.schedule_stream, prepared)
        self.assertEqual(module.created, [])
        self.assertTrue(scheduler._war_barrier_enabled)

    def test_cuda_and_hip_alias_redraw_stays_bounded(self) -> None:
        for cuda, hip in ((True, False), (False, True)):
            with self.subTest(cuda=cuda, hip=hip):
                scheduler, module, _ = self._scheduler(handles=(11,) * 65)
                with (
                    patch.object(native, "use_mlx", return_value=False),
                    patch.object(native, "is_cuda", return_value=cuda),
                    patch.object(native, "_is_hip", hip),
                ):
                    scheduler.init_schedule_stream()
                self.assertEqual(len(module.created), 65)
                self.assertIs(scheduler.schedule_stream, module.created[-1])

    def test_cpu_synchronize_remains_noop_and_loop_reuses_stream(self) -> None:
        scheduler, module, events = self._scheduler(device="cpu")
        with (
            patch.object(native, "use_mlx", return_value=False),
            patch.object(native, "is_cuda", return_value=False),
        ):
            scheduler.init_schedule_stream()
        stream = scheduler.schedule_stream
        self.assertIsNone(stream.synchronize())
        self.assertFalse(isinstance(stream.synchronize, Mock))
        self._run_event_loop(scheduler, events, mlx=False)
        self.assertIs(scheduler.schedule_stream, stream)
        self.assertEqual(len(module.created), 1)

    def test_mlx_allocates_no_stream_and_preserves_dispatch(self) -> None:
        scheduler, module, events = self._scheduler(device="mps")
        with patch.object(native, "use_mlx", return_value=True):
            scheduler.init_schedule_stream()
        self.assertFalse(hasattr(scheduler, "schedule_stream"))
        self._run_event_loop(scheduler, events, mlx=True)
        self.assertEqual(module.created, [])
        self.assertEqual(events, ["watch-install", "watch-serving", "dispatch"])

    def test_cpu_regressions_do_not_initialize_cuda(self) -> None:
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
