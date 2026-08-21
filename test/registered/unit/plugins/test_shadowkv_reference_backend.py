"""Reference-GPU backend integration tests with generation-owned request state."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from predkv_inference.shadowkv.configuration import ShadowKVConfig
from predkv_inference.shadowkv.oracle import gqa_attention
from predkv_inference.shadowkv.reference_backend import (
    ReferenceGPUAttentionBackend,
    ReferenceRuntimeManager,
)
from predkv_inference.shadowkv.state import RequestStateStatus
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Mode:
    def __init__(self, name):
        self.name = name

    def is_extend(self):
        return self.name in {"EXTEND", "TARGET_VERIFY"}

    def is_decode(self):
        return self.name == "DECODE"

    def is_mixed(self):
        return self.name == "MIXED"


class _Delegate:
    def __init__(self):
        self.calls = []

    def forward(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        self.calls.append((q, k, v, layer, forward_batch, save_kv_cache, kwargs))
        return q.clone()


class _RequestPool:
    def __init__(self):
        self.req_generation = torch.tensor([0, 1], dtype=torch.int64)


class TestShadowKVReferenceBackend(CustomTestCase):
    def setUp(self):
        self.config = ShadowKVConfig(
            mode="reference-gpu",
            rank=4,
            chunk_size=2,
            sparse_budget=4,
            outlier_chunks=1,
            local_chunks=1,
            generated_tail_tokens=2,
        )
        self.pool = _RequestPool()
        self.manager = ReferenceRuntimeManager(
            self.config,
            self.pool,
            num_layers=1,
            kv_heads=2,
            head_dim=4,
        )
        self.allocation = SimpleNamespace(
            request_id="request-a", request_pool_index=1, generation=1
        )
        self.manager.on_request_admitted(self.allocation)
        self.delegate = _Delegate()
        self.backend = ReferenceGPUAttentionBackend(
            self.delegate,
            SimpleNamespace(to_dict=lambda: {}),
            self.config,
            self.manager,
        )
        self.layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=4,
            tp_k_head_num=2,
            qk_head_dim=4,
        )
        generator = torch.Generator().manual_seed(20260821)
        self.q = torch.randn(7, 16, generator=generator)
        self.pre = torch.randn(7, 2, 4, generator=generator)
        self.k = self.pre + 0.125
        self.v = torch.randn(7, 2, 4, generator=generator)

    def _prefill_batch(self):
        return SimpleNamespace(
            forward_mode=_Mode("EXTEND"),
            extend_seq_lens_cpu=[7],
            extend_prefix_lens_cpu=[0],
            req_pool_indices=torch.tensor([1]),
            rids=["request-a"],
        )

    def test_prefill_decode_growth_and_terminal_cleanup(self):
        dense_output = self.backend.forward(
            self.q,
            self.k,
            self.v,
            self.layer,
            self._prefill_batch(),
            pre_rope_key=self.pre,
        )
        torch.testing.assert_close(dense_output, self.q, rtol=0, atol=0)
        identity = self.manager.store.active_identity(1)
        state = self.manager.store.state(identity)
        self.assertEqual(state.status, RequestStateStatus.INITIALIZED)
        self.assertTrue(state.layers[0].exact_short_prompt)

        decode_q = torch.randn(1, 16, generator=torch.Generator().manual_seed(8))
        decode_k = torch.randn(1, 2, 4, generator=torch.Generator().manual_seed(9))
        decode_v = torch.randn(1, 2, 4, generator=torch.Generator().manual_seed(10))
        decode_batch = SimpleNamespace(
            forward_mode=_Mode("DECODE"),
            req_pool_indices=torch.tensor([1]),
            rids=["request-a"],
        )
        observed = self.backend.forward(
            decode_q,
            decode_k,
            decode_v,
            self.layer,
            decode_batch,
            pre_rope_key=decode_k.clone(),
        )
        expected = gqa_attention(
            decode_q.reshape(1, 4, 4).numpy(),
            torch.cat(
                (self.k.transpose(0, 1), decode_k.transpose(0, 1)), dim=1
            ).numpy(),
            torch.cat(
                (self.v.transpose(0, 1), decode_v.transpose(0, 1)), dim=1
            ).numpy(),
        )
        np.testing.assert_allclose(
            observed.reshape(1, 4, 4).numpy(), expected, rtol=2e-5, atol=2e-5
        )
        self.assertEqual(state.generated_tokens, 1)
        self.assertEqual(len(self.delegate.calls), 2)

        event = SimpleNamespace(
            allocation=self.allocation,
            reason=SimpleNamespace(value="completion"),
        )
        self.manager.on_request_terminal(event)
        self.assertEqual(state.status, RequestStateStatus.RELEASED)
        self.assertFalse(state.layers)

    def test_unsupported_forward_and_aliased_key_fail_before_delegate(self):
        unsupported = self._prefill_batch()
        unsupported.forward_mode = _Mode("TARGET_VERIFY")
        with self.assertRaisesRegex(RuntimeError, "unsupported ForwardMode"):
            self.backend.forward(
                self.q,
                self.k,
                self.v,
                self.layer,
                unsupported,
                pre_rope_key=self.pre,
            )
        with self.assertRaisesRegex(RuntimeError, "aliases"):
            self.backend.forward(
                self.q,
                self.k,
                self.v,
                self.layer,
                self._prefill_batch(),
                pre_rope_key=self.k,
            )
        self.assertFalse(self.delegate.calls)


if __name__ == "__main__":
    unittest.main()
