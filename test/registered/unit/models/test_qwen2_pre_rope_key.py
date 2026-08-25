"""CPU tests for Qwen2's optional pre-RoPE key handoff."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import sglang.srt.layers.radix_attention as radix_attention_module
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.qwen2 import Qwen2Attention

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Projection:
    def __init__(self, qkv):
        self.qkv = qkv
        self.original = None

    def __call__(self, hidden_states):
        del hidden_states
        projected = self.qkv.clone()
        self.original = projected.clone()
        return projected, None


class _Rotary:
    def __call__(self, positions, query, key):
        del positions
        return query + 10, key + 100


class _OutputProjection:
    def __call__(self, value):
        return value * 2, None


class _RecordingBackend:
    def __init__(self, requires_pre_rope_key):
        self.required = requires_pre_rope_key
        self.calls = []

    def requires_pre_rope_key(self, layer):
        del layer
        return self.required

    def forward(self, query, key, value, layer, forward_batch, save_kv_cache, **kwargs):
        del layer, forward_batch, save_kv_cache
        self.calls.append(
            SimpleNamespace(query=query, key=key, value=value, kwargs=kwargs)
        )
        return query


class TestQwen2PreRopeKey(CustomTestCase):
    def _run(self, required):
        qkv = torch.arange(36, dtype=torch.float32).view(2, 18)
        projection = _Projection(qkv)
        harness = SimpleNamespace(
            q_size=6,
            kv_size=6,
            qkv_proj=projection,
            rotary_emb=_Rotary(),
            attn=RadixAttention(2, 3, 1.0, num_kv_heads=2, layer_id=0),
            o_proj=_OutputProjection(),
        )
        backend = _RecordingBackend(required)
        with (
            patch.object(
                radix_attention_module,
                "get_tc_piecewise_forward_context",
                return_value=None,
            ),
            patch.object(
                radix_attention_module, "get_attn_backend", return_value=backend
            ),
        ):
            output = Qwen2Attention.forward(
                harness,
                positions=torch.arange(2),
                hidden_states=torch.zeros((2, 6)),
                forward_batch=SimpleNamespace(forward_mode=ForwardMode.DECODE),
            )
        return output, projection, backend

    def test_dense_path_has_no_capture_or_output_change(self):
        output, projection, backend = self._run(False)
        self.assertEqual(backend.calls[0].kwargs, {})
        self.assertTrue(torch.equal(output, (projection.original[:, :6] + 10) * 2))

    def test_requesting_backend_receives_unrotated_nonaliasing_key(self):
        output, projection, backend = self._run(True)
        call = backend.calls[0]
        expected = projection.original[:, 6:12].view(2, 2, 3)
        self.assertTrue(torch.equal(call.kwargs["pre_rope_key"], expected))
        self.assertTrue(torch.equal(call.key, expected + 100))
        self.assertNotEqual(call.kwargs["pre_rope_key"].data_ptr(), call.key.data_ptr())
        self.assertTrue(torch.equal(output, (projection.original[:, :6] + 10) * 2))


if __name__ == "__main__":
    unittest.main()
