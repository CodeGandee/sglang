"""Unit tests for model-runner layer discovery."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.distributed.utils import get_pp_indices
from sglang.srt.model_executor.model_runner_components.layer_setup import (
    compute_attention_and_moe_layers,
    resolve_layer_indices,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestComputeAttentionAndMoeLayers(unittest.TestCase):
    def test_deepseek_mla_registers_mha_companion(self):
        attn_mqa = SimpleNamespace()
        attn_mha = SimpleNamespace()
        layer_model = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    self_attn=SimpleNamespace(attn_mqa=attn_mqa, attn_mha=attn_mha)
                )
            ]
        )

        attention_layers, _, _, _, mha_companion_layers = (
            compute_attention_and_moe_layers(layer_model)
        )

        self.assertEqual(attention_layers, [attn_mqa])
        self.assertEqual(mha_companion_layers, [attn_mha])
        self.assertNotIn("_pcg_mha_companion", vars(attn_mqa))


def _config(*, layers=78, nextn=1, architecture="GlmMoeDsaForCausalLM", loop_num=1):
    return SimpleNamespace(
        num_hidden_layers=layers,
        num_attention_layers=layers,
        num_nextn_predict_layers=nextn,
        hf_config=SimpleNamespace(architectures=[architecture], loop_num=loop_num),
    )


class TestResolveDFlashLayerIndices(unittest.TestCase):
    """Keep separate trained drafting distinct from target-bundled MTP layers."""

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def _target_partitions(self, algorithm):
        expected = {
            1: ((0, 78),),
            2: ((0, 39), (39, 78)),
            3: ((0, 26), (26, 52), (52, 78)),
            4: ((0, 19), (19, 38), (38, 58), (58, 78)),
        }
        with patch.dict(os.environ):
            os.environ.pop("SGLANG_PP_LAYER_PARTITION", None)
            for pp_size, ranges in expected.items():
                for rank, wanted in enumerate(ranges):
                    with self.subTest(pp_size=pp_size, rank=rank):
                        start, end = get_pp_indices(78, rank, pp_size)
                        self.assertEqual((start, end), wanted)
                        config = _config()
                        info = resolve_layer_indices(
                            model=SimpleNamespace(start_layer=start, end_layer=end),
                            model_config=config,
                            is_draft_worker=False,
                            spec_algorithm=algorithm,
                        )
                        self.assertEqual((info.start_layer, info.end_layer), wanted)
                        self.assertEqual(info.num_effective_layers, end - start)
                        self.assertEqual(config.num_hidden_layers, 78)
                        self.assertEqual(config.num_nextn_predict_layers, 1)

    def test_default_dflash_target_ignores_unused_bundled_mtp_layers(self):
        args = ServerArgs(model_path="dummy", speculative_algorithm="DFLASH")
        self.assertEqual(args.speculative_dflash_implementation, "ours")
        self._target_partitions(
            SpeculativeAlgorithm.from_string(args.speculative_algorithm)
        )

    def test_ordinary_target_keeps_identical_layer_ranges(self):
        self._target_partitions(SpeculativeAlgorithm.NONE)

    def test_other_speculative_algorithms_keep_mtp_partition_restriction(self):
        for algorithm in SpeculativeAlgorithm:
            if algorithm.is_none() or algorithm.is_dflash():
                continue
            with self.subTest(algorithm=algorithm.name):
                with self.assertRaisesRegex(AssertionError, "PP.*MTP"):
                    resolve_layer_indices(
                        model=SimpleNamespace(start_layer=39, end_layer=78),
                        model_config=_config(),
                        is_draft_worker=False,
                        spec_algorithm=algorithm,
                    )

    def test_pp1_true_mtp_target_and_draft_counts_remain_separate(self):
        config = _config()
        target = resolve_layer_indices(
            model=SimpleNamespace(),
            model_config=config,
            is_draft_worker=False,
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
        )
        draft = resolve_layer_indices(
            model=SimpleNamespace(),
            model_config=config,
            is_draft_worker=True,
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
        )
        self.assertEqual(
            (target.start_layer, target.end_layer, target.num_effective_layers),
            (0, 78, 78),
        )
        self.assertEqual(
            (draft.start_layer, draft.end_layer, draft.num_effective_layers), (0, 1, 1)
        )
        stages = resolve_layer_indices(
            model=SimpleNamespace(num_stages=2),
            model_config=_config(nextn=3),
            is_draft_worker=True,
            spec_algorithm=SpeculativeAlgorithm.FROZEN_KV_MTP,
        )
        self.assertEqual(
            (stages.start_layer, stages.end_layer, stages.num_effective_layers),
            (0, 2, 2),
        )

    def test_separate_dflash_draft_keeps_its_own_six_layers(self):
        for nextn in (None, 0):
            with self.subTest(nextn=nextn):
                info = resolve_layer_indices(
                    model=SimpleNamespace(),
                    model_config=_config(
                        layers=6, nextn=nextn, architecture="DFlash2DraftModel"
                    ),
                    is_draft_worker=True,
                    spec_algorithm=SpeculativeAlgorithm.DFLASH,
                )
                self.assertEqual(
                    (info.start_layer, info.end_layer, info.num_effective_layers),
                    (0, 6, 6),
                )

    def test_true_mtp_architecture_overrides_and_loop_counts_stay_intact(self):
        for architecture in (
            "MiMoV2MTP",
            "Step3p5MTP",
            "InklingForConditionalGenerationMTP",
        ):
            with self.subTest(architecture=architecture):
                info = resolve_layer_indices(
                    model=SimpleNamespace(),
                    model_config=_config(nextn=3, architecture=architecture),
                    is_draft_worker=True,
                    spec_algorithm=SpeculativeAlgorithm.EAGLE,
                )
                self.assertEqual(
                    (info.start_layer, info.end_layer, info.num_effective_layers),
                    (0, 1, 1),
                )
        info = resolve_layer_indices(
            model=SimpleNamespace(start_layer=17, end_layer=41),
            model_config=_config(loop_num=2),
            is_draft_worker=False,
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
        )
        self.assertEqual(
            (info.start_layer, info.end_layer, info.num_effective_layers), (17, 41, 48)
        )


if __name__ == "__main__":
    unittest.main()
