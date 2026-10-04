"""Exercise optional fused GEMM eligibility and actual native caller dispatch."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.kernels.ops.gemm import fused_a_gemm as gemm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _projection(shape=(16, 256), dtype=torch.bfloat16, *, packed=False):
    projection = ReplicatedLinear.__new__(ReplicatedLinear)
    torch.nn.Module.__init__(projection)
    projection.register_parameter(
        "weight_packed" if packed else "weight",
        torch.nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False),
    )
    projection.register_parameter("bias", None)
    projection.skip_bias_add = False
    projection.quant_method = SimpleNamespace(apply=Mock())
    return projection


def _actual_prepare_qkv_latent():
    """Compile the actual native caller, avoiding distributed model construction."""
    source = (
        Path(__file__).resolve().parents[4] / "python/sglang/srt/models/deepseek_v2.py"
    )
    module = ast.parse(source.read_text())
    attention = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "DeepseekV2AttentionMLA"
    )
    method = next(
        node
        for node in attention.body
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_qkv_latent"
    )
    compiled = ast.Module(
        body=[ast.parse("from __future__ import annotations").body[0], method],
        type_ignores=[],
    )
    namespace = {
        "get_exec": lambda: SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=False)
        ),
        "fused_a_gemm_weight_eligible": gemm.fused_a_gemm_weight_eligible,
        "linear_with_fused_a_gemm": gemm.linear_with_fused_a_gemm,
    }
    exec(compile(ast.fix_missing_locations(compiled), str(source), "exec"), namespace)
    return namespace["prepare_qkv_latent"]


def _attention(projection):
    return SimpleNamespace(
        q_lora_rank=2,
        has_fused_proj=True,
        # compressed_tensors packed format is not covered by the old AWQ-only
        # constructor flag; the actual eligibility helper must inspect weight.
        is_packed_weight=False,
        fused_qkv_a_proj_with_mqa=projection,
        _use_min_latency_fused_a_gemm=None,
        fused_a_gemm_backend="auto",
    )


class TestFusedAGemmEligibility(unittest.TestCase):
    def test_actual_packed_projection_is_ineligible(self):
        projection = _projection((16, 64), torch.int32, packed=True)
        self.assertIsInstance(projection, ReplicatedLinear)
        self.assertFalse(hasattr(projection, "weight"))
        with (
            patch.object(gemm, "_IS_CUDA", True),
            patch.object(gemm, "_DEVICE_SM", 100),
        ):
            self.assertFalse(gemm.fused_a_gemm_weight_eligible(projection))

    def test_missing_and_non_tensor_ordinary_weights_are_ineligible(self):
        for projection in (torch.nn.Module(), SimpleNamespace(weight=None)):
            with self.subTest(projection=type(projection).__name__):
                self.assertFalse(gemm.fused_a_gemm_weight_eligible(projection))

    def test_ordinary_weight_shape_dtype_and_platform_conditions_are_preserved(self):
        cases = (
            ((16, 256), torch.bfloat16, True, 90, True),
            ((32, 512), torch.bfloat16, True, 100, True),
            ((15, 256), torch.bfloat16, True, 100, False),
            ((16, 255), torch.bfloat16, True, 100, False),
            ((16, 256), torch.float16, True, 100, False),
            ((16, 256), torch.int8, True, 100, False),
            ((16, 256), torch.bfloat16, False, 100, False),
            ((16, 256), torch.bfloat16, True, 80, False),
            ((16,), torch.bfloat16, True, 100, False),
            ((16, 256, 1), torch.bfloat16, True, 100, False),
        )
        for shape, dtype, cuda, sm, expected in cases:
            with self.subTest(shape=shape, dtype=dtype, cuda=cuda, sm=sm):
                projection = _projection(shape, dtype)
                with (
                    patch.object(gemm, "_IS_CUDA", cuda),
                    patch.object(gemm, "_DEVICE_SM", sm),
                ):
                    self.assertEqual(
                        gemm.fused_a_gemm_weight_eligible(projection), expected
                    )

    def test_actual_caller_keeps_packed_default_quant_method_dispatch(self):
        projection = _projection((16, 64), torch.int32, packed=True)
        attention = _attention(projection)
        hidden = torch.zeros((1, 256), dtype=torch.bfloat16)
        expected = torch.arange(16, dtype=torch.bfloat16).reshape(1, 16)
        projection.quant_method.apply.return_value = expected
        prepare = _actual_prepare_qkv_latent()
        with (
            patch.object(gemm, "_IS_CUDA", True),
            patch.object(gemm, "_DEVICE_SM", 100),
            patch.object(
                gemm,
                "dsv3_fused_a_gemm",
                side_effect=AssertionError("packed weights must keep quant dispatch"),
            ) as fused,
        ):
            actual = prepare(attention, hidden, None)
        self.assertIs(actual, expected)
        projection.quant_method.apply.assert_called_once_with(projection, hidden, None)
        fused.assert_not_called()
        self.assertFalse(attention._use_min_latency_fused_a_gemm)

    def test_actual_caller_preserves_eligible_bf16_fused_dispatch(self):
        projection = _projection()
        attention = _attention(projection)
        hidden = torch.zeros((1, 256), dtype=torch.bfloat16)
        expected = torch.arange(16, dtype=torch.bfloat16).reshape(1, 16)
        with (
            patch.object(gemm, "_IS_CUDA", True),
            patch.object(gemm, "_DEVICE_SM", 100),
            patch.object(gemm, "dsv3_fused_a_gemm", return_value=expected) as fused,
        ):
            actual = _actual_prepare_qkv_latent()(attention, hidden, None)
        self.assertIs(actual, expected)
        self.assertTrue(attention._use_min_latency_fused_a_gemm)
        projection.quant_method.apply.assert_not_called()
        arguments, keywords = fused.call_args
        self.assertIs(arguments[0], hidden)
        self.assertTrue(torch.equal(arguments[1], projection.weight.T))
        self.assertEqual(keywords, {"backend": "auto"})
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
