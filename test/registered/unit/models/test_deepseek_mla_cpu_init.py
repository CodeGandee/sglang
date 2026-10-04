"""Check actual CPU-fused MLA initialization without allocating a model."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    forward_mla_fused_rope_cpu as cpu_forward,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _projection(dtype=torch.bfloat16, *, packed=False):
    projection = ReplicatedLinear.__new__(ReplicatedLinear)
    torch.nn.Module.__init__(projection)
    projection.register_parameter(
        "weight_packed" if packed else "weight",
        torch.nn.Parameter(
            torch.empty((2, 2), dtype=dtype, device="meta"), requires_grad=False
        ),
    )
    projection.quant_method = SimpleNamespace(block_quant=False)
    return projection


def _attention(projection, *, packed=False):
    return SimpleNamespace(
        has_fused_proj=True,
        is_packed_weight=packed,
        fused_qkv_a_proj_with_mqa=projection,
        q_b_proj=_projection(),
        quant_method="unchanged",
    )


def _initialize(attention, *, cpu, amx=False):
    with (
        patch.object(cpu_forward, "_is_cpu", cpu),
        patch.object(cpu_forward, "_is_cpu_amx_available", amx),
    ):
        cpu_forward.DeepseekMLACpuForwardMixin.init_mla_fused_rope_cpu_forward(
            attention
        )


class TestMLACpuInitialization(unittest.TestCase):
    def test_packed_non_cpu_projection_never_reads_ordinary_weight(self):
        projection = _projection(torch.int32, packed=True)
        self.assertIsInstance(projection, ReplicatedLinear)
        attention = _attention(projection)
        original = ReplicatedLinear.__getattribute__

        def refuse_weight_read(instance, name):
            if name == "weight":
                raise AssertionError("non-CPU initializer must not inspect weight")
            return original(instance, name)

        with patch.object(ReplicatedLinear, "__getattribute__", refuse_weight_read):
            _initialize(attention, cpu=False, amx=True)
        self.assertFalse(attention.qkv_proj_with_rope_is_int8)
        self.assertFalse(attention.qkv_proj_with_rope_is_fp8)
        self.assertIsNone(attention.weight_block_size)
        self.assertEqual(attention.quant_method, "unchanged")

    def test_packed_cpu_projection_without_ordinary_weight_is_ineligible(self):
        projection = _projection(torch.int32, packed=True)
        self.assertFalse(hasattr(projection, "weight"))
        attention = _attention(projection)
        _initialize(attention, cpu=True)
        self.assertFalse(attention.qkv_proj_with_rope_is_int8)
        self.assertFalse(attention.qkv_proj_with_rope_is_fp8)
        self.assertIsNone(attention.weight_block_size)

    def test_existing_packed_flag_short_circuits_dtype_checks(self):
        attention = _attention(_projection(torch.int32, packed=True), packed=True)
        _initialize(attention, cpu=True)
        self.assertFalse(attention.qkv_proj_with_rope_is_int8)
        self.assertFalse(attention.qkv_proj_with_rope_is_fp8)

    def test_absent_fused_projection_needs_no_weights(self):
        attention = SimpleNamespace(has_fused_proj=False, is_packed_weight=False)
        _initialize(attention, cpu=True, amx=True)
        self.assertFalse(attention.qkv_proj_with_rope_is_int8)
        self.assertFalse(attention.qkv_proj_with_rope_is_fp8)
        self.assertIsNone(attention.weight_block_size)

    def test_cpu_ordinary_dtype_flags_and_amx_metadata_are_preserved(self):
        for dtype, int8, fp8 in (
            (torch.int8, True, False),
            (torch.float8_e4m3fn, False, True),
            (torch.bfloat16, False, False),
        ):
            for amx in (False, True):
                with self.subTest(dtype=dtype, amx=amx):
                    attention = _attention(_projection(dtype))
                    pack = Mock(return_value="packed absorb weights")
                    with patch.object(cpu_forward, "PackWeightMethod", pack):
                        _initialize(attention, cpu=True, amx=amx)
                    self.assertEqual(attention.qkv_proj_with_rope_is_int8, int8)
                    self.assertEqual(attention.qkv_proj_with_rope_is_fp8, fp8)
                    self.assertIsNone(attention.weight_block_size)
                    if amx:
                        pack.assert_called_once_with(
                            weight_names=["w_kc", "w_vc"],
                            transpose_dims=[[1, 2], [1, 2]],
                        )
                        self.assertEqual(
                            attention.quant_method, "packed absorb weights"
                        )
                    else:
                        pack.assert_not_called()
                        self.assertEqual(attention.quant_method, "unchanged")

    def test_cpu_fp8_block_quant_metadata_is_preserved(self):
        attention = _attention(_projection(torch.float8_e4m3fn))
        for projection in (attention.fused_qkv_a_proj_with_mqa, attention.q_b_proj):
            projection.quant_method = SimpleNamespace(
                block_quant=True,
                quant_config=SimpleNamespace(weight_block_size=[128, 128]),
            )
        with patch.object(cpu_forward, "PackWeightMethod", return_value="packed"):
            _initialize(attention, cpu=True, amx=True)
        self.assertTrue(attention.qkv_proj_with_rope_is_fp8)
        self.assertEqual(attention.weight_block_size, [128, 128])

    def test_cpu_fp8_mismatched_block_quant_metadata_still_refuses(self):
        attention = _attention(_projection(torch.float8_e4m3fn))
        attention.fused_qkv_a_proj_with_mqa.quant_method.block_quant = True
        with patch.object(cpu_forward, "PackWeightMethod", return_value="packed"):
            with self.assertRaises(AssertionError):
                _initialize(attention, cpu=True, amx=True)


if __name__ == "__main__":
    unittest.main()
