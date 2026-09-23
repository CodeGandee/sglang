"""CPU contracts for the optional B1 DSA decode graph metadata banks."""

import unittest
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.layers.attention import dsa_backend as dsa
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _CPUBackend(dsa.DeepseekSparseAttnBackend):
    """Exercise the native graph planner with CPU tensors and no GPU kernels."""

    def __init__(self):
        self.device = torch.device("cpu")
        self.real_page_size = 1
        self.hisparse_coordinator = None
        self.speculative_num_draft_tokens = 0
        self.use_fused_topk = False
        self.dsa_topk_backend = SimpleNamespace(should_use_topk_v2=lambda: False)
        self.dsa_index_topk = 4
        self.dsa_decode_impl = "tilelang"
        self.dsa_prefill_impl = "tilelang"
        self.use_mha = False
        self.req_to_token = torch.arange(32, dtype=torch.int32).reshape(2, 16)
        self._arange_buf = torch.arange(16, dtype=torch.int32)

    def set_dsa_prefill_impl(self, forward_batch=None):
        self.use_mha = False


def _batch(req_index: int, seq_len: int) -> ForwardBatch:
    return ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=1,
        input_ids=torch.tensor([1], dtype=torch.int32),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int32),
        req_pool_indices=torch.tensor([req_index], dtype=torch.int32),
        out_cache_loc=torch.tensor([0], dtype=torch.int32),
        seq_lens_sum=seq_len,
        positions=torch.tensor([seq_len - 1], dtype=torch.int64),
    )


def _writable_addresses(metadata: dsa.DSAMetadata) -> set[int]:
    shared_immutable = {"cu_seqlens_q", "dsa_cu_seqlens_q"}
    return {
        value.data_ptr()
        for field in fields(metadata)
        if field.name not in shared_immutable
        if isinstance(value := getattr(metadata, field.name), torch.Tensor)
    }


class TestDSADecodeMetadataBanks(unittest.TestCase):
    def setUp(self):
        self.is_cuda = patch.object(dsa, "is_cuda", return_value=False)
        self.is_cuda.start()
        self.addCleanup(self.is_cuda.stop)
        self.backend = _CPUBackend()

    def test_default_graph_state_still_refreshes_its_single_bank(self):
        backend = self.backend
        backend.init_cuda_graph_state(1, 1)
        default_state = backend.decode_cuda_graph_metadata
        backend.init_forward_metadata_out_graph(_batch(0, 3), in_capture=True)
        default_metadata = backend.forward_metadata

        backend.init_forward_metadata_out_graph(_batch(1, 5), in_capture=False)

        self.assertIs(backend.decode_cuda_graph_metadata, default_state)
        self.assertIs(backend.forward_metadata, default_metadata)
        self.assertEqual(default_metadata.cache_seqlens_int32.tolist(), [5])
        self.assertEqual(default_metadata.page_table_1[0].tolist(), list(range(16, 32)))

        backend.prepare_decode_metadata_bank(_batch(0, 3))
        self.assertIs(backend.decode_cuda_graph_metadata, default_state)
        self.assertIs(backend.forward_metadata, default_metadata)

    def test_alternating_banks_keep_writable_storage_and_values_separate(self):
        backend = self.backend
        real = backend.prepare_decode_metadata_bank(_batch(0, 3))
        hint = backend.prepare_decode_metadata_bank(_batch(1, 5))
        self.assertIsNot(real.graph_state, hint.graph_state)
        self.assertIsNot(real.metadata, hint.metadata)
        self.assertFalse(
            _writable_addresses(real.metadata) & _writable_addresses(hint.metadata)
        )
        # The backend-owned arange is read-only and may be shared across banks.
        self.assertEqual(
            real.metadata.dsa_cu_seqlens_q.data_ptr(),
            hint.metadata.dsa_cu_seqlens_q.data_ptr(),
        )

        backend.refresh_decode_metadata_bank(real, _batch(1, 7))
        backend.refresh_decode_metadata_bank(hint, _batch(0, 9))
        backend.refresh_decode_metadata_bank(real, _batch(0, 11))

        self.assertEqual(real.metadata.cache_seqlens_int32.tolist(), [11])
        self.assertEqual(hint.metadata.cache_seqlens_int32.tolist(), [9])
        self.assertEqual(real.metadata.page_table_1[0].tolist(), list(range(16)))
        self.assertEqual(hint.metadata.page_table_1[0].tolist(), list(range(16)))
        with backend.activate_decode_metadata_bank(hint) as metadata:
            self.assertIs(metadata, hint.metadata)
            self.assertIs(backend.forward_metadata, hint.metadata)
            with backend.activate_decode_metadata_bank(real):
                self.assertIs(backend.forward_metadata, real.metadata)
            self.assertIs(backend.forward_metadata, hint.metadata)
        self.assertFalse(hasattr(backend, "forward_metadata"))
        self.assertFalse(hasattr(backend, "decode_cuda_graph_metadata"))

    def test_prepare_refresh_and_active_body_restore_after_failure(self):
        backend = self.backend
        backend.init_cuda_graph_state(1, 1)
        ordinary_state = backend.decode_cuda_graph_metadata
        backend.init_forward_metadata_out_graph(_batch(0, 2), in_capture=True)
        ordinary_metadata = backend.forward_metadata
        backend.use_mha = True
        backend.dsa_prefill_impl = "flashmla_sparse"

        with (
            patch.object(
                backend, "init_cuda_graph_state", side_effect=RuntimeError("allocate")
            ),
            self.assertRaisesRegex(RuntimeError, "allocate"),
        ):
            backend.prepare_decode_metadata_bank(_batch(0, 3))
        self.assertIs(backend.decode_cuda_graph_metadata, ordinary_state)
        self.assertIs(backend.forward_metadata, ordinary_metadata)
        self.assertTrue(backend.use_mha)
        self.assertEqual(backend.dsa_prefill_impl, "flashmla_sparse")

        with (
            patch.object(
                backend,
                "init_forward_metadata_out_graph",
                side_effect=RuntimeError("prepare"),
            ),
            self.assertRaisesRegex(RuntimeError, "prepare"),
        ):
            backend.prepare_decode_metadata_bank(_batch(0, 3))
        self.assertIs(backend.decode_cuda_graph_metadata, ordinary_state)
        self.assertIs(backend.forward_metadata, ordinary_metadata)
        self.assertTrue(backend.use_mha)
        self.assertEqual(backend.dsa_prefill_impl, "flashmla_sparse")

        bank = backend.prepare_decode_metadata_bank(_batch(0, 3))
        with (
            patch.object(
                backend,
                "init_forward_metadata_out_graph",
                side_effect=RuntimeError("refresh"),
            ),
            self.assertRaisesRegex(RuntimeError, "refresh"),
        ):
            backend.refresh_decode_metadata_bank(bank, _batch(1, 4))
        self.assertIs(backend.decode_cuda_graph_metadata, ordinary_state)
        self.assertIs(backend.forward_metadata, ordinary_metadata)
        self.assertTrue(backend.use_mha)
        self.assertEqual(backend.dsa_prefill_impl, "flashmla_sparse")

        with (
            self.assertRaisesRegex(RuntimeError, "model"),
            backend.activate_decode_metadata_bank(bank),
        ):
            raise RuntimeError("model")
        self.assertIs(backend.decode_cuda_graph_metadata, ordinary_state)
        self.assertIs(backend.forward_metadata, ordinary_metadata)
        self.assertTrue(backend.use_mha)
        self.assertEqual(backend.dsa_prefill_impl, "flashmla_sparse")

    def test_rejects_non_decode_and_over_capacity_without_changing_backend(self):
        backend = self.backend
        batch = _batch(0, 17)
        with self.assertRaisesRegex(ValueError, "exceeds request mapping"):
            backend.prepare_decode_metadata_bank(batch)
        batch = _batch(0, 4)
        batch.forward_mode = ForwardMode.EXTEND
        with self.assertRaisesRegex(ValueError, "B1 DECODE"):
            backend.prepare_decode_metadata_bank(batch)
        self.assertFalse(hasattr(backend, "decode_cuda_graph_metadata"))


if __name__ == "__main__":
    unittest.main()
