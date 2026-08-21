"""CPU tests for Llama's optional pre-RoPE key handoff."""

import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import sglang.srt.layers.radix_attention as radix_attention_module
import sglang.srt.models.llama as llama_module
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.llama import LlamaAttention

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Projection:
    def __init__(self, qkv: torch.Tensor):
        self.qkv = qkv
        self.original = None

    def __call__(self, hidden_states):
        projected = self.qkv.clone()
        self.original = projected.clone()
        return projected, None


class _Rotary:
    def __call__(self, positions, query, key):
        query.add_(10)
        key.add_(100)
        return query, key


class _OutputProjection:
    def __call__(self, value):
        return value * 2, None


class _RecordingBackend:
    def __init__(self, *, requires_pre_rope_key: bool):
        self._requires_pre_rope_key = requires_pre_rope_key
        self.calls = []

    def requires_pre_rope_key(self, layer):
        return self._requires_pre_rope_key

    def forward(
        self,
        query,
        key,
        value,
        attention_layer,
        forward_batch,
        save_kv_cache,
        **kwargs,
    ):
        self.calls.append(
            SimpleNamespace(query=query, key=key, value=value, kwargs=kwargs)
        )
        return query


def _make_harness(qkv: torch.Tensor):
    projection = _Projection(qkv)
    harness = SimpleNamespace(
        q_size=6,
        kv_size=6,
        qkv_proj=projection,
        rotary_emb=_Rotary(),
        attn=RadixAttention(
            num_heads=2,
            head_dim=3,
            scaling=1.0,
            num_kv_heads=2,
            layer_id=0,
        ),
        o_proj=_OutputProjection(),
    )
    harness.forward_prepare_native = MethodType(
        LlamaAttention.forward_prepare_native, harness
    )
    harness.forward_prepare_npu = MethodType(
        LlamaAttention.forward_prepare_npu, harness
    )
    return harness, projection


class TestLlamaPreRopeKey(CustomTestCase):
    def _run(self, *, requires_pre_rope_key: bool):
        qkv = torch.arange(36, dtype=torch.float32).view(2, 18)
        harness, projection = _make_harness(qkv)
        backend = _RecordingBackend(
            requires_pre_rope_key=requires_pre_rope_key
        )
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)

        with (
            patch.object(llama_module, "_is_npu", False),
            patch.object(
                radix_attention_module,
                "get_tc_piecewise_forward_context",
                return_value=None,
            ),
            patch.object(
                radix_attention_module, "get_attn_backend", return_value=backend
            ),
        ):
            output = LlamaAttention.forward(
                harness,
                positions=torch.arange(2),
                hidden_states=torch.zeros((2, 6)),
                forward_batch=forward_batch,
            )

        return output, projection, backend

    def test_dense_path_retains_no_auxiliary_key_and_preserves_output(self):
        output, projection, backend = self._run(requires_pre_rope_key=False)

        expected = (projection.original[:, :6] + 10) * 2
        self.assertTrue(torch.equal(output, expected))
        self.assertEqual(backend.calls[0].kwargs, {})

    def test_requesting_backend_receives_shaped_unrotated_key(self):
        output, projection, backend = self._run(requires_pre_rope_key=True)

        call = backend.calls[0]
        pre_rope_key = call.kwargs["pre_rope_key"]
        expected_pre_rope = projection.original[:, 6:12].view(2, 2, 3)
        expected_rotated = expected_pre_rope + 100
        expected_output = (projection.original[:, :6] + 10) * 2

        self.assertEqual(pre_rope_key.shape, (2, 2, 3))
        self.assertTrue(torch.equal(pre_rope_key, expected_pre_rope))
        self.assertTrue(torch.equal(call.key, expected_rotated))
        self.assertNotEqual(pre_rope_key.data_ptr(), call.key.data_ptr())
        self.assertTrue(torch.equal(output, expected_output))


if __name__ == "__main__":
    unittest.main()
