import types
import unittest
from unittest.mock import patch

from sglang.srt.layers import communicator as comm
from sglang.srt.layers.communicator import LayerCommunicator, ScatterMode
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fake_communicator(*, is_pp_stage_end=False, is_last_layer=False):
    return types.SimpleNamespace(
        _speculative_algo=None,
        layer_scatter_modes=types.SimpleNamespace(mlp_mode=ScatterMode.TP_ATTN_FULL),
        is_last_layer=is_last_layer,
        is_pp_stage_end=is_pp_stage_end,
        allow_reduce_scatter=True,
        _communicate_summable_tensor_pair_fn=comm.CommunicateSummableTensorPairFn._trivial,
        _context=types.SimpleNamespace(tp_size=4),
    )


class TestFuseMlpAllReduceGate(CustomTestCase):
    """Hybrid EP+TP must not fuse the post-experts all-reduce away.

    The fused residual+LN reduces over a single group, but with moe_ep_size > 1
    and moe_tp_size > 1 the post-experts reduction spans two disjoint groups
    (_MOE_EP then _MOE_TP) and should_skip_post_experts_all_reduce() drops both
    once fusion is published. The result is activations reduced over only half
    the peers -- wrong output, no crash. Observed as garbage completions on
    Qwen3-30B-A3B with --tp-size 4 --ep-size 2.
    """

    def _should_fuse(
        self, *, moe_ep_size, moe_tp_size, is_pp_stage_end=False, is_last_layer=False
    ):
        forward_batch = types.SimpleNamespace(
            input_ids=types.SimpleNamespace(shape=(8,))
        )
        with (
            patch.object(comm, "is_enable_moe_cp_allgather", return_value=False),
            patch.object(comm, "apply_flashinfer_allreduce_fusion", return_value=True),
            patch.object(
                comm,
                "get_attn_tp_context",
                return_value=types.SimpleNamespace(input_scattered=False),
            ),
            get_parallel().override(
                moe_ep_size=moe_ep_size, moe_tp_size=moe_tp_size, tp_size=4
            ),
        ):
            return LayerCommunicator.should_fuse_mlp_allreduce_with_next_layer(
                _fake_communicator(
                    is_pp_stage_end=is_pp_stage_end, is_last_layer=is_last_layer
                ),
                forward_batch,
            )

    def test_hybrid_ep_tp_does_not_fuse(self):
        self.assertFalse(self._should_fuse(moe_ep_size=2, moe_tp_size=2))

    def test_pure_tp_still_fuses(self):
        self.assertTrue(self._should_fuse(moe_ep_size=1, moe_tp_size=4))

    def test_pure_ep_still_fuses(self):
        self.assertTrue(self._should_fuse(moe_ep_size=4, moe_tp_size=1))

    def test_pp_stage_end_does_not_defer_reduction(self):
        self.assertFalse(
            self._should_fuse(moe_ep_size=1, moe_tp_size=4, is_pp_stage_end=True)
        )

    def test_global_final_and_nextn_gate_remains_closed(self):
        for stage_end in (False, True):
            with self.subTest(stage_end=stage_end):
                self.assertFalse(
                    self._should_fuse(
                        moe_ep_size=1,
                        moe_tp_size=4,
                        is_pp_stage_end=stage_end,
                        is_last_layer=True,
                    )
                )

    def test_stage_marker_preserves_input_scattered_reduce_scatter(self):
        with (
            patch.object(comm, "dsa_use_prefill_cp", return_value=False),
            patch.object(comm, "mla_use_prefill_cp", return_value=False),
            patch.object(
                comm,
                "get_attn_tp_context",
                return_value=types.SimpleNamespace(input_scattered=True),
            ),
        ):
            for stage_end in (False, True):
                for global_final in (False, True):
                    with self.subTest(stage_end=stage_end, global_final=global_final):
                        self.assertEqual(
                            LayerCommunicator.should_use_reduce_scatter(
                                _fake_communicator(
                                    is_pp_stage_end=stage_end,
                                    is_last_layer=global_final,
                                ),
                                types.SimpleNamespace(),
                            ),
                            not global_final,
                        )

    def test_stage_marker_preserves_cp_reduce_scatter(self):
        for cp_gate in ("dsa_use_prefill_cp", "mla_use_prefill_cp"):
            with (
                patch.object(comm, "dsa_use_prefill_cp", return_value=False),
                patch.object(comm, "mla_use_prefill_cp", return_value=False),
                patch.object(comm, cp_gate, return_value=True),
            ):
                for stage_end in (False, True):
                    with self.subTest(cp_gate=cp_gate, stage_end=stage_end):
                        self.assertTrue(
                            LayerCommunicator.should_use_reduce_scatter(
                                _fake_communicator(is_pp_stage_end=stage_end),
                                types.SimpleNamespace(),
                            )
                        )


if __name__ == "__main__":
    unittest.main()
