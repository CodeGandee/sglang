"""Check native graph auxiliary owner retention and replay notification on CPU."""

import unittest

import torch

from sglang.srt.layers.aux_hidden_states import (
    AuxHiddenStatePacker,
    capture_aux_packers,
    publish_aux_replay,
    set_aux_replay_observer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class AuxGraphReplayTests(unittest.TestCase):
    def tearDown(self) -> None:
        set_aux_replay_observer(None)

    def _packer(self) -> AuxHiddenStatePacker:
        packer = AuxHiddenStatePacker(1)
        packer.append(torch.ones((8, 4), dtype=torch.bfloat16))
        return packer

    def test_completed_owner_retained_only_inside_capture(self) -> None:
        self._packer()
        with capture_aux_packers() as retained:
            packer = self._packer()
        self.assertEqual(retained.packers, (packer,))

    def test_duplicate_and_incomplete_owner_refuse_before_any_callback(self) -> None:
        calls: list[AuxHiddenStatePacker] = []
        set_aux_replay_observer(calls.append)
        packer = self._packer()
        for owners in ((packer, packer), (packer, AuxHiddenStatePacker(1))):
            with self.subTest(owners=len(owners)):
                with self.assertRaises((RuntimeError, AssertionError)):
                    publish_aux_replay(owners)
                self.assertEqual(calls, [])
        publish_aux_replay((packer,))
        self.assertEqual(calls, [packer])

    def test_nested_capture_refuses_without_disturbing_outer_owner(self) -> None:
        with capture_aux_packers() as retained:
            with self.assertRaises(RuntimeError):
                with capture_aux_packers():
                    pass
            packer = self._packer()
        self.assertEqual(retained.packers, (packer,))
