"""Readable ShadowKV decode differentials on CPU or B200."""

import os
import unittest

import numpy as np
import torch

from predkv_inference.shadowkv.configuration import ShadowKVConfig
from predkv_inference.shadowkv.oracle import (
    apply_llama_rope,
    build_prefill_reference,
    gqa_attention,
    reconstruct_keys,
    select_sparse_positions,
)
from predkv_inference.shadowkv.reference_runtime import (
    ShadowKVReferenceError,
    apply_torch_llama_rope,
    build_torch_prefill_state,
    readable_torch_decode,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestShadowKVReferenceDecode(CustomTestCase):
    def setUp(self):
        self.device = os.getenv("SHADOWKV_TEST_DEVICE", "cpu")
        self.config = ShadowKVConfig(
            rank=8,
            chunk_size=2,
            sparse_budget=4,
            outlier_chunks=1,
            local_chunks=1,
        )

    def _case(self, seed, decode_position):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        pre = torch.randn(
            12, 2, 4, generator=generator, device=self.device, dtype=torch.float32
        )
        prompt_positions = torch.arange(12, device=self.device)[:, None].expand(-1, 2)
        rotated = apply_torch_llama_rope(
            pre.transpose(0, 1), prompt_positions.T
        ).transpose(0, 1)
        values = torch.randn(
            12, 2, 4, generator=generator, device=self.device, dtype=torch.float32
        )
        query = torch.randn(
            1, 4, 4, generator=generator, device=self.device, dtype=torch.float32
        )
        current_pre_key = torch.randn(
            2, 1, 4, generator=generator, device=self.device, dtype=torch.float32
        )
        current_positions = torch.full(
            (2, 1), decode_position, device=self.device, dtype=torch.long
        )
        current_key = apply_torch_llama_rope(current_pre_key, current_positions)
        current_value = torch.randn(
            2, 1, 4, generator=generator, device=self.device, dtype=torch.float32
        )

        state = build_torch_prefill_state(pre, rotated, values, self.config)
        observed_output, observed_kv = readable_torch_decode(
            state,
            query,
            self.config,
            generated_keys=current_key,
            generated_values=current_value,
        )

        pre_np = pre.cpu().numpy()
        rotated_np = rotated.cpu().numpy()
        values_np = values.cpu().numpy()
        query_np = query.cpu().numpy()
        expected_state = build_prefill_reference(pre_np, rotated_np, self.config)
        expected_selection = select_sparse_positions(
            expected_state,
            query_np,
            self.config,
            generated_positions=np.array([decode_position]),
        )
        selected = expected_selection.selected_positions
        selected_keys = reconstruct_keys(expected_state.factors, selected)
        selected_keys = apply_llama_rope(selected_keys, selected)
        selected_values = np.stack(
            [values_np[selected[head], head] for head in range(2)]
        )
        expected_keys = np.concatenate(
            (
                rotated_np[expected_state.local_positions].transpose(1, 0, 2),
                np.stack(
                    [
                        rotated_np[expected_state.outlier_positions[head], head]
                        for head in range(2)
                    ]
                ),
                selected_keys,
                current_key.cpu().numpy(),
            ),
            axis=1,
        )
        expected_values = np.concatenate(
            (
                values_np[expected_state.local_positions].transpose(1, 0, 2),
                np.stack(
                    [
                        values_np[expected_state.outlier_positions[head], head]
                        for head in range(2)
                    ]
                ),
                selected_values,
                current_value.cpu().numpy(),
            ),
            axis=1,
        )
        expected_output = gqa_attention(query_np, expected_keys, expected_values)

        np.testing.assert_array_equal(
            observed_kv.selection.selected_chunk_indices.cpu().numpy(),
            expected_selection.selected_chunk_indices,
        )
        np.testing.assert_allclose(
            observed_kv.keys.cpu().numpy(), expected_keys, rtol=2e-5, atol=2e-5
        )
        np.testing.assert_allclose(
            observed_kv.values.cpu().numpy(), expected_values, rtol=0, atol=0
        )
        np.testing.assert_allclose(
            observed_output.cpu().numpy(), expected_output, rtol=2e-5, atol=2e-5
        )

    def test_operation_differentials_across_layers_and_positions(self):
        for layer_seed, position in ((3, 12), (11, 19), (29, 127)):
            with self.subTest(layer_seed=layer_seed, position=position):
                self._case(layer_seed, position)

    def test_nonfinite_decode_fails_explicitly(self):
        generator = torch.Generator(device=self.device).manual_seed(7)
        pre = torch.randn(7, 2, 4, generator=generator, device=self.device)
        state = build_torch_prefill_state(pre, pre, pre, self.config)
        query = torch.zeros(1, 4, 4, device=self.device)
        query[0, 0, 0] = torch.inf
        with self.assertRaisesRegex(ShadowKVReferenceError, "non-finite"):
            readable_torch_decode(state, query, self.config)

    def test_short_prompt_route_matches_exact_attention(self):
        generator = torch.Generator(device=self.device).manual_seed(71)
        pre = torch.randn(7, 2, 4, generator=generator, device=self.device)
        positions = torch.arange(7, device=self.device)[:, None].expand(-1, 2)
        rotated = apply_torch_llama_rope(pre.transpose(0, 1), positions.T).transpose(
            0, 1
        )
        values = torch.randn(7, 2, 4, generator=generator, device=self.device)
        query = torch.randn(1, 4, 4, generator=generator, device=self.device)
        state = build_torch_prefill_state(pre, rotated, values, self.config)

        observed, sparse_kv = readable_torch_decode(state, query, self.config)
        expected = gqa_attention(
            query.cpu().numpy(),
            rotated.transpose(0, 1).cpu().numpy(),
            values.transpose(0, 1).cpu().numpy(),
        )

        self.assertTrue(sparse_kv.exact_short_prompt)
        self.assertIsNone(sparse_kv.selection)
        np.testing.assert_allclose(
            observed.cpu().numpy(), expected, rtol=2e-5, atol=2e-5
        )


if __name__ == "__main__":
    unittest.main()
