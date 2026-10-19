"""CPU contracts for prefill graph replay frame delegation.

The breakable graph backend refuses a nested replay whose frame differs from
the enclosing target scope's published frame. These tests pin the prefill
runner helper that resolves that frame without CUDA or a model.
"""

import unittest
from typing import cast

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    delegated_replay_frame,
    get_graph_forward_batch,
    graph_forward_batch_scope,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _frame() -> ForwardBatch:
    return cast(ForwardBatch, object())


class TestPrefillReplayFrameDelegation(unittest.TestCase):
    def test_published_live_frame_is_delegated(self) -> None:
        live = _frame()
        static = _frame()
        with graph_forward_batch_scope(live):
            frame = delegated_replay_frame(live, static)
            self.assertIs(frame, live)
            with graph_forward_batch_scope(frame):
                self.assertIs(get_graph_forward_batch(), live)
            self.assertIs(get_graph_forward_batch(), live)
        self.assertIsNone(get_graph_forward_batch())

    def test_foreign_frame_is_still_refused(self) -> None:
        live = _frame()
        static = _frame()
        foreign = _frame()
        with graph_forward_batch_scope(foreign):
            frame = delegated_replay_frame(live, static)
            self.assertIs(frame, static)
            with (
                self.assertRaisesRegex(RuntimeError, "foreign target forward frame"),
                graph_forward_batch_scope(frame),
            ):
                pass
        self.assertIsNone(get_graph_forward_batch())

    def test_no_enclosing_scope_keeps_static_frame(self) -> None:
        live = _frame()
        static = _frame()
        frame = delegated_replay_frame(live, static)
        self.assertIs(frame, static)
        with graph_forward_batch_scope(frame):
            self.assertIs(get_graph_forward_batch(), static)
        self.assertIsNone(get_graph_forward_batch())

    def test_missing_live_frame_keeps_static_frame(self) -> None:
        published = _frame()
        static = _frame()
        with graph_forward_batch_scope(published):
            frame = delegated_replay_frame(None, static)
            self.assertIs(frame, static)
        self.assertIsNone(get_graph_forward_batch())


if __name__ == "__main__":
    unittest.main()
