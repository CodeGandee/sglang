"""Exercise actual A-projection fusion and replicated loading on CPU tensors."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from transformers import PretrainedConfig

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.parameter import BasevLLMParameter
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.models.deepseek_common import deepseek_weight_loader as loader
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _CpuLoader(loader.DeepseekV2WeightLoaderMixin, torch.nn.Module):
    """Bind real fusion/loading to small parameters; omit model post-processing."""

    def __init__(self, fields, *, quantized=True):
        torch.nn.Module.__init__(self)
        self.config = PretrainedConfig(
            n_routed_experts=0, q_lora_rank=2, num_hidden_layers=1
        )
        self.quant_config = (
            CompressedTensorsConfig({}, [], "pack-quantized", {}, [])
            if quantized
            else None
        )
        self.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
        self.num_fused_shared_experts = 0
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        attention = torch.nn.Module()
        self.model.layers[0].self_attn = attention
        projection = ReplicatedLinear.__new__(ReplicatedLinear)
        torch.nn.Module.__init__(projection)
        attention.fused_qkv_a_proj_with_mqa = projection
        self.projection = projection
        for name, data in fields.items():
            projection.register_parameter(
                name,
                BasevLLMParameter(data=data, weight_loader=projection.weight_loader),
            )
        self.post_load_weights = Mock()

    def load(self, fields, reverse=False):
        weights = []
        for suffix, (q, kv) in fields.items():
            pair = [
                (f"model.layers.0.self_attn.q_a_proj.{suffix}", q),
                (f"model.layers.0.self_attn.kv_a_proj_with_mqa.{suffix}", kv),
            ]
            weights.extend(reversed(pair) if reverse else pair)
        # Native async loading obtains a CUDA device. CPU fixtures use its actual
        # synchronous route while retaining real fusion and parameter loaders.
        with patch.object(loader, "should_async_load", return_value=False):
            self.do_load_weights(weights)


class TestAProjectionWeightShapeFusion(unittest.TestCase):
    def test_checkpoint_descriptors_merge_output_rows_in_both_arrival_orders(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                fixture = _CpuLoader(
                    {"weight_shape": torch.zeros(2, dtype=torch.int64)}
                )
                q = torch.tensor([2048, 6144], dtype=torch.int64)
                kv = torch.tensor([576, 6144], dtype=torch.int64)
                fixture.load({"weight_shape": (q, kv)}, reverse)
                self.assertTrue(
                    torch.equal(
                        fixture.projection.weight_shape,
                        torch.tensor([2624, 6144], dtype=torch.int64),
                    )
                )
                self.assertEqual(fixture.projection.weight_shape.dtype, torch.int64)
                self.assertTrue(torch.equal(q, torch.tensor([2048, 6144])))
                self.assertTrue(torch.equal(kv, torch.tensor([576, 6144])))
                fixture.post_load_weights.assert_called_once()

    def test_packed_weights_and_scales_keep_exact_row_concatenation(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                fixture = _CpuLoader(
                    {
                        "weight_shape": torch.zeros(2, dtype=torch.int64),
                        "weight_packed": torch.zeros((5, 2), dtype=torch.int32),
                        "weight_scale": torch.zeros((5, 1), dtype=torch.bfloat16),
                    }
                )
                q = torch.tensor([[11, 12], [21, 22]], dtype=torch.int32)
                kv = torch.tensor([[31, 32], [41, 42], [51, 52]], dtype=torch.int32)
                qs = torch.tensor([[1], [2]], dtype=torch.bfloat16)
                kvs = torch.tensor([[3], [4], [5]], dtype=torch.bfloat16)
                fixture.load(
                    {
                        "weight_packed": (q, kv),
                        "weight_shape": (torch.tensor([2, 8]), torch.tensor([3, 8])),
                        "weight_scale": (qs, kvs),
                    },
                    reverse,
                )
                self.assertTrue(
                    torch.equal(fixture.projection.weight_packed, torch.cat([q, kv]))
                )
                self.assertTrue(
                    torch.equal(fixture.projection.weight_scale, torch.cat([qs, kvs]))
                )
                self.assertTrue(
                    torch.equal(fixture.projection.weight_shape, torch.tensor([5, 8]))
                )

    def test_ordinary_bf16_weights_keep_exact_concatenation(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                fixture = _CpuLoader(
                    {"weight": torch.zeros((5, 4), dtype=torch.bfloat16)},
                    quantized=False,
                )
                q = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
                kv = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4) + 20
                fixture.load({"weight": (q, kv)}, reverse)
                self.assertTrue(
                    torch.equal(fixture.projection.weight, torch.cat([q, kv]))
                )

    def test_scalar_field_keeps_existing_q_selection_in_both_orders(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                fixture = _CpuLoader({"weight_scale": torch.zeros(1)})
                fixture.load(
                    {"weight_scale": (torch.tensor(2.0), torch.tensor(3.0))}, reverse
                )
                self.assertTrue(
                    torch.equal(fixture.projection.weight_scale, torch.tensor([2.0]))
                )

    def test_invalid_shape_descriptors_are_refused_before_parameter_write(self):
        valid = torch.tensor([2, 8], dtype=torch.int64)
        malformed = (
            (valid.to(torch.float32), torch.tensor([3, 8])),
            (valid, torch.tensor([3, 8], dtype=torch.int32)),
            (valid, torch.tensor([[3, 8]])),
            (valid, torch.tensor([3, 8, 9])),
            (valid, torch.tensor([3, 16])),
            (valid, torch.tensor([0, 8])),
            (torch.tensor([-1, 8]), torch.tensor([3, 8])),
            (torch.tensor([2, 0]), torch.tensor([3, 0])),
        )
        for pair in malformed:
            for reverse in (False, True):
                with self.subTest(pair=pair, reverse=reverse):
                    fixture = _CpuLoader(
                        {"weight_shape": torch.full((2,), -1, dtype=torch.int64)}
                    )
                    with self.assertRaises(ValueError):
                        fixture.load({"weight_shape": pair}, reverse)
                    self.assertTrue(
                        torch.equal(
                            fixture.projection.weight_shape, torch.tensor([-1, -1])
                        )
                    )
                    fixture.post_load_weights.assert_not_called()

    def test_replication_shape_assertion_remains_for_ordinary_weights(self):
        fixture = _CpuLoader({"weight": torch.zeros((5, 4))}, quantized=False)
        with self.assertRaises(AssertionError):
            fixture.load({"weight": (torch.ones((2, 3)), torch.ones((3, 3)))})
        self.assertTrue(torch.equal(fixture.projection.weight, torch.zeros((5, 4))))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
