"""Dense-parity plugin tests with the installed root distribution."""

import os
import unittest
from types import SimpleNamespace

import torch

from predkv_inference.shadowkv.runtime_backend import DenseParityAttentionBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _Delegate:
    needs_cpu_seq_lens = False

    def __init__(self):
        self.calls = []

    def forward(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        self.calls.append((q, k, v, layer, forward_batch, save_kv_cache, kwargs))
        return q + k.repeat_interleave(2, dim=1) + v.repeat_interleave(2, dim=1)


class TestShadowKVDenseParityBackend(CustomTestCase):
    def setUp(self):
        self.delegate = _Delegate()
        self.backend = DenseParityAttentionBackend(
            self.delegate, SimpleNamespace(to_dict=lambda: {})
        )
        self.layer = SimpleNamespace(tp_k_head_num=2, qk_head_dim=4)
        self.forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: False)
        )
        device = os.getenv("SHADOWKV_TEST_DEVICE", "cpu")
        self.q = torch.arange(32, dtype=torch.float32, device=device).reshape(2, 4, 4)
        self.k = torch.arange(16, dtype=torch.float32, device=device).reshape(2, 2, 4)
        self.v = self.k + 3
        self.pre_rope = self.k.clone() - 7

    def test_delegated_output_is_bitwise_unchanged_and_key_is_not_retained(self):
        expected = self.delegate.forward(
            self.q, self.k, self.v, self.layer, self.forward_batch
        )
        self.delegate.calls.clear()

        observed = self.backend.forward(
            self.q,
            self.k,
            self.v,
            self.layer,
            self.forward_batch,
            pre_rope_key=self.pre_rope,
        )

        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
        self.assertEqual(self.backend.pre_rope_observations, 1)
        self.assertEqual(self.backend.last_pre_rope_shape, (2, 2, 4))
        self.assertIsNone(self.backend.retained_pre_rope_key)
        self.assertEqual(self.backend.shadowkv_runtime_state_bytes, 0)
        self.assertTrue(self.backend._reported_first_forward)
        self.assertEqual(self.delegate.calls[0][-1], {})
        self.assertFalse(self.backend.needs_cpu_seq_lens)

    def test_missing_malformed_or_aliased_pre_rope_key_is_rejected(self):
        cases = (
            ({}, "expected"),
            ({"pre_rope_key": torch.empty(2, 1, 4)}, "shape mismatch"),
            ({"pre_rope_key": self.k}, "aliases"),
        )
        for kwargs, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(RuntimeError, message),
            ):
                self.backend.forward(
                    self.q,
                    self.k,
                    self.v,
                    self.layer,
                    self.forward_batch,
                    **kwargs,
                )


if __name__ == "__main__":
    unittest.main()
