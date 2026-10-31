"""Verify group-wide backend selection and fail-closed lazy initialization."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch
from sglang.srt.distributed.device_communicators import triton_symm_mem_ag as ag
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMultimemInitialization(TestCase):
    def setUp(self):
        self.contexts = ExitStack()
        self.addCleanup(self.contexts.close)
        self.group = SimpleNamespace(
            world_size=8, rank_in_group=0, cpu_group=object(), device_group=object()
        )
        for target, kwargs in (
            ("sglang.srt.distributed.get_tp_group", {"return_value": self.group}),
            (
                "sglang.srt.runtime_context.get_parallel",
                {"return_value": SimpleNamespace(nnodes=1)},
            ),
            ("torch.cuda.is_available", {"return_value": True}),
            ("torch.cuda.get_device_capability", {"return_value": (9, 0)}),
            ("torch.cuda.is_current_stream_capturing", {"return_value": False}),
            ("torch.distributed.all_reduce", {}),
            ("torch.distributed.barrier", {}),
        ):
            mock = self.contexts.enter_context(patch(target, **kwargs))
            setattr(self, target.rsplit(".", 1)[1], mock)
        self.x = torch.ones((1, 8), dtype=torch.bfloat16)
        self.state = SimpleNamespace(
            max_token_num=128,
            world_size=8,
            hidden_dim=64,
            symm_mem_hdl=SimpleNamespace(multicast_ptr=16),
        )
        self.create = self.contexts.enter_context(
            patch.object(ag, "create_state", return_value=self.state)
        )
        self.native = self.contexts.enter_context(
            patch("sglang.srt.distributed.tensor_model_parallel_all_gather")
        )
        self.multimem = self.contexts.enter_context(
            patch.object(ag, "all_gather_inner")
        )

    def test_sm80_uses_nccl_without_symmetric_allocation(self):
        self.get_device_capability.return_value = (8, 0)
        gather = ag.MultimemAllGatherer(128)
        self.assertIs(gather(self.x), self.native.return_value)
        self.create.assert_not_called()
        self.barrier.assert_not_called()
        self.all_reduce.assert_called_once()

    def test_one_unsupported_peer_disables_multimem_on_supported_rank(self):
        self.all_reduce.side_effect = lambda flags, **kwargs: flags.zero_()
        gather = ag.MultimemAllGatherer(128)
        self.assertIs(gather(self.x), self.native.return_value)
        self.create.assert_not_called()

    def test_supported_initialization_joins_before_allocation_and_runs_once(self):
        events = []
        self.barrier.side_effect = lambda **kwargs: events.append("join")
        self.create.side_effect = lambda **kwargs: (
            events.append("allocate") or self.state
        )
        gather = ag.MultimemAllGatherer(128)
        for _ in range(2):
            self.assertIs(gather(self.x), self.multimem.return_value)
        self.assertEqual(events, ["join", "allocate"])
        self.assertEqual(self.all_reduce.call_count, 2)
        self.assertIs(gather._state, self.state)

    def test_missing_multicast_on_one_peer_selects_nccl_everywhere(self):
        def agree(flags, **kwargs):
            if self.all_reduce.call_count == 2:
                flags.zero_()

        self.all_reduce.side_effect = agree
        gather = ag.MultimemAllGatherer(128)
        self.assertIs(gather(self.x), self.native.return_value)
        self.assertIsNone(gather._state)
        self.multimem.assert_not_called()

    def test_rendezvous_failure_never_falls_back_independently(self):
        failure = RuntimeError("peer rendezvous failed")
        self.create.side_effect = failure
        gather = ag.MultimemAllGatherer(128)
        with self.assertRaisesRegex(RuntimeError, "abort the TP group") as caught:
            gather(self.x)
        self.assertIs(caught.exception.__cause__, failure)
        self.native.assert_not_called()
        self.multimem.assert_not_called()

    def test_join_failure_never_starts_rendezvous_or_nccl_fallback(self):
        self.barrier.side_effect = RuntimeError("peer did not join")
        gather = ag.MultimemAllGatherer(128)
        with self.assertRaisesRegex(RuntimeError, "peer did not join"):
            gather(self.x)
        self.create.assert_not_called()
        self.native.assert_not_called()

    def test_capture_defers_allocation_until_next_eager_call(self):
        gather = ag.MultimemAllGatherer(128)
        self.is_current_stream_capturing.return_value = True
        self.assertIs(gather(self.x), self.native.return_value)
        self.assertIs(gather._state, gather._UNINIT)
        self.create.assert_not_called()
        self.barrier.assert_not_called()
        self.is_current_stream_capturing.return_value = False
        self.assertIs(gather(self.x), self.multimem.return_value)

    def test_disabled_and_single_rank_need_no_capability_collective(self):
        ag.MultimemAllGatherer(128, enabled=False)
        self.group.world_size = 1
        gather = ag.MultimemAllGatherer(128)
        self.assertIs(gather(self.x), self.native.return_value)
        self.get_device_capability.assert_not_called()
        self.all_reduce.assert_not_called()
        self.create.assert_not_called()

    def test_ineligible_dtype_uses_native_gather(self):
        gather = ag.MultimemAllGatherer(128)
        self.assertIs(gather(self.x.float()), self.native.return_value)
        self.create.assert_not_called()
        self.barrier.assert_not_called()
