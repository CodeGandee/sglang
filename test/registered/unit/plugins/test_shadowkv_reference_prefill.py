"""Readable ShadowKV prefill differentials against the independent NumPy oracle."""

import os
import unittest

import numpy as np
import torch

from predkv_inference.shadowkv.configuration import ShadowKVConfig
from predkv_inference.shadowkv.oracle import build_prefill_reference
from predkv_inference.shadowkv.reference_runtime import build_torch_prefill_state
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestShadowKVReferencePrefill(CustomTestCase):
    def setUp(self):
        self.device = os.getenv("SHADOWKV_TEST_DEVICE", "cpu")
        self.config = ShadowKVConfig(
            rank=8,
            chunk_size=2,
            sparse_budget=4,
            outlier_chunks=1,
            local_chunks=1,
        )
        generator = torch.Generator(device=self.device).manual_seed(20260821)
        self.pre = torch.randn(
            12,
            2,
            4,
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        self.rotated = (
            self.pre + torch.linspace(0, 0.1, 12, device=self.device)[:, None, None]
        )
        self.values = self.pre * 0.25

    def test_prefill_invariants_match_independent_oracle(self):
        observed = build_torch_prefill_state(
            self.pre, self.rotated, self.values, self.config
        )
        expected = build_prefill_reference(
            self.pre.cpu().numpy(), self.rotated.cpu().numpy(), self.config
        )

        self.assertFalse(observed.exact_short_prompt)
        self.assertEqual(tuple(observed.factors.u.shape), (12, 8))
        self.assertEqual(tuple(observed.factors.sv.shape), (2, 8, 4))
        np.testing.assert_array_equal(
            observed.outlier_chunk_indices.cpu().numpy(),
            expected.outlier_chunk_indices,
        )
        np.testing.assert_array_equal(
            observed.landmark_chunk_indices.cpu().numpy(),
            expected.landmark_chunk_indices,
        )
        np.testing.assert_allclose(
            observed.landmarks.cpu().numpy(), expected.landmarks, rtol=1e-6, atol=1e-6
        )
        reconstructed = torch.einsum(
            "tr,hrd->thd", observed.factors.u, observed.factors.sv
        )
        torch.testing.assert_close(reconstructed, self.pre, rtol=2e-5, atol=2e-5)
        self.assertEqual(tuple(observed.exact_local_keys.shape), (2, 2, 4))
        self.assertEqual(tuple(observed.exact_outlier_keys.shape), (2, 2, 4))

    def test_short_prompt_keeps_exact_gpu_state_without_factors(self):
        observed = build_torch_prefill_state(
            self.pre[:7], self.rotated[:7], self.values[:7], self.config
        )

        self.assertTrue(observed.exact_short_prompt)
        self.assertIsNone(observed.factors)
        torch.testing.assert_close(
            observed.prompt_rotated_keys,
            self.rotated[:7].transpose(0, 1),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
