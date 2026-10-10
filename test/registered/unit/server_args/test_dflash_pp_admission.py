"""Exercise final DFlash PP admission through real native ServerArgs methods.

Dummy model construction keeps model loading and hardware initialization out of
this CPU contract; argument validation and worker selection remain real.
"""

import unittest

import torch

from sglang.srt.arg_groups.speculative_hook import _handle_dflash
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _args(**overrides: object) -> ServerArgs:
    values = {
        "model_path": "dummy",
        "served_model_name": "dummy",
        "device": "cuda",
        "tp_size": 1,
        "pp_size": 2,
        "disable_overlap_schedule": True,
        "speculative_algorithm": "DFLASH",
        "speculative_draft_model_path": "unused-draft",
        "speculative_draft_attention_backend": "fa4",
        "speculative_dflash_block_size": 8,
        "speculative_num_draft_tokens": 8,
        "max_running_requests": 2,
        "chunked_prefill_size": 2048,
        "page_size": 64,
    }
    values.update(overrides)
    return ServerArgs(**values)


class TestDFlashPPAdmission(CustomTestCase):
    """Keep maintained PP qualification and other native restrictions distinct."""

    def tearDown(self) -> None:
        self.assertFalse(torch.cuda.is_initialized())

    def test_default_and_explicit_ours_admit_pp2_pp3_pp4(self) -> None:
        from predkv_extensions.dflash2.worker import DFlashWorker

        for pp_size in (2, 3, 4):
            for selector in ({}, {"speculative_dflash_implementation": "ours"}):
                with self.subTest(pp_size=pp_size, selector=selector):
                    args = _args(pp_size=pp_size, **selector)
                    _handle_dflash(args)
                    args.check_server_args()
                    self.assertIs(
                        SpeculativeAlgorithm.DFLASH.create_worker(args), DFlashWorker
                    )
                    self.assertEqual(args.pp_size, pp_size)
                    self.assertEqual(args.pp_async_batch_depth, 0)
                    self.assertTrue(args.disable_overlap_schedule)
                    self.assertEqual(args.speculative_dflash_implementation, "ours")

    def test_ordinary_pp_control_remains_admitted(self) -> None:
        for pp_size in (2, 3, 4):
            with self.subTest(pp_size=pp_size):
                args = _args(pp_size=pp_size, speculative_algorithm=None)
                args.check_server_args()
                self.assertIsNone(args.speculative_algorithm)

    def test_original_keeps_pp1_worker_and_rejects_pp2_pp3_pp4(self) -> None:
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        args = _args(
            pp_size=1,
            disable_overlap_schedule=False,
            speculative_dflash_implementation="original",
        )
        _handle_dflash(args)
        args.check_server_args()
        self.assertIs(SpeculativeAlgorithm.DFLASH.create_worker(args), DFlashWorkerV2)
        for pp_size in (2, 3, 4):
            with self.subTest(pp_size=pp_size):
                args = _args(
                    pp_size=pp_size, speculative_dflash_implementation="original"
                )
                with self.assertRaisesRegex(AssertionError, "Pipeline parallelism"):
                    args.check_server_args()
                with self.assertRaisesRegex(ValueError, "original DFLASH.*pp_size"):
                    _handle_dflash(args)
                self.assertEqual(args.speculative_dflash_implementation, "original")

    def test_other_speculative_algorithms_remain_rejected_in_pp(self) -> None:
        for algorithm in SpeculativeAlgorithm:
            if algorithm.is_none() or algorithm.is_dflash():
                continue
            with self.subTest(algorithm=algorithm.name):
                args = _args(speculative_algorithm=algorithm.name)
                with self.assertRaisesRegex(AssertionError, "Pipeline parallelism"):
                    args.check_server_args()
                self.assertEqual(args.speculative_algorithm, algorithm.name)

    def test_ours_pp_rejects_overlap_and_async_depth(self) -> None:
        with self.assertRaisesRegex(AssertionError, "Pipeline parallelism"):
            _args(disable_overlap_schedule=False).check_server_args()
        args = _args(pp_async_batch_depth=1)
        with self.assertRaisesRegex(AssertionError, "Pipeline parallelism"):
            args.check_server_args()
        with self.assertRaisesRegex(ValueError, "wait-based handoff"):
            _handle_dflash(args)
        self.assertEqual(args.pp_async_batch_depth, 1)

    def test_supported_pp_still_runs_later_native_validation(self) -> None:
        cases = (
            ({"min_free_slots_delay": 1}, AssertionError, "min-free-slots-delay"),
            ({"chunked_prefill_size": 2049}, AssertionError, "divisible by page_size"),
            ({"enable_mixed_chunk": True}, AssertionError, "enable_mixed_chunk"),
            ({"enable_lora": True}, ValueError, "LoRA.*NGRAM"),
        )
        for overrides, error, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(error, message):
                    _args(**overrides).check_server_args()


if __name__ == "__main__":
    unittest.main()
