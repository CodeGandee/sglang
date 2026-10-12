"""Run native grouped HiSparse entrypoints through their real GLM callback.

CPU ownership and final kernel stubs cover dispatch and callback binding only;
they do not qualify FlashMLA numerical execution or CUDA storage behavior.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from predkv_extensions.glm.hisparse.attention import GLMGroupedAttention
from predkv_extensions.glm.hisparse.coordinator import GLMHiSparseCoordinator
from sglang.srt.layers.attention import dsa_backend as dsa
from sglang.srt.mem_cache import memory_pool
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDSAHiSparseDispatch(unittest.TestCase):
    """Check both native entrypoints preserve callback ownership and operands."""

    def _run(self, *, extend: bool, split_query: bool = True) -> None:
        query_count = 2 if extend else 1
        rows = 2 * query_count
        selected = torch.arange(rows * 3, dtype=torch.int32).reshape(rows, 3)
        q_nope = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 2, 2)
        q_rope = q_nope + 100
        producer_cache = torch.ones((64, 1, 656), dtype=torch.uint8)
        fetched_cache = torch.zeros_like(producer_cache)
        lengths = torch.full((rows,), 3, dtype=torch.int32)
        metadata = dsa.DSAMetadata(
            page_size=64,
            cache_seqlens_int32=lengths,
            max_seq_len_q=query_count,
            max_seq_len_k=8,
            cu_seqlens_q=torch.tensor([0, query_count, rows]),
            cu_seqlens_k=torch.tensor([0, 4, 10]),
            page_table_1=None,
            real_page_table=torch.zeros((2, 1), dtype=torch.int32),
            dsa_cache_seqlens_int32=lengths,
            dsa_cu_seqlens_q=torch.arange(rows + 1),
            dsa_cu_seqlens_k=torch.arange(rows + 1) * 3,
            dsa_extend_seq_lens_list=[query_count, query_count],
            dsa_seqlens_expanded=torch.tensor([5, 7]).repeat_interleave(query_count),
        )
        fetches = []
        consumed_layers = []
        released_native = []
        released_fetch = []
        produced = []
        stream = object()
        committed = torch.tensor([4, 6], dtype=torch.int32)
        round_state = SimpleNamespace(
            owners=(
                SimpleNamespace(scratch=torch.tensor([40, 41])),
                SimpleNamespace(scratch=torch.tensor([42, 43])),
            ),
            block_size=query_count,
            committed=committed,
        )
        co = GLMHiSparseCoordinator.__new__(GLMHiSparseCoordinator)
        co._generic_hisparse = SimpleNamespace(
            storage=SimpleNamespace(max_entries=16),
            bound_request_identities=lambda: (
                {"slot": 3, "core_generation": 7},
                {"slot": 5, "core_generation": 9},
            ),
            acquire_layer=lambda local_layer, actual_stream: SimpleNamespace(
                release=lambda value: released_native.append(
                    (local_layer, actual_stream, value)
                )
            ),
        )
        co.mem_pool_device = SimpleNamespace(
            start_layer=8,
            layer_num=1,
            full_to_hisparse_device_index_mapping=torch.arange(64) + 40,
        )
        co.mem_pool_host = SimpleNamespace(kv_buffer=(torch.zeros((64, 656)),))
        co.req_to_host_pool = torch.zeros((6, 16), dtype=torch.int64)
        co.top_k = 3
        co.wait_for_pending_backup = lambda: None
        co._capacity_agreement = None
        co._target_kv = SimpleNamespace(
            _round=round_state if extend else None,
            active=extend,
            record_layer=consumed_layers.append,
        )

        def fetch(**operands):
            fetches.append(operands)
            return SimpleNamespace(
                cache=fetched_cache,
                locations=operands["selections"] + 20,
                release=released_fetch.append,
            )

        co._group_attention = GLMGroupedAttention(co, SimpleNamespace(fetch=fetch))
        backend = dsa.DeepseekSparseAttnBackend.__new__(dsa.DeepseekSparseAttnBackend)
        backend.hisparse_coordinator = co
        backend.forward_metadata = metadata
        backend.dsa_decode_impl = "flashmla_kv"
        backend.dsa_prefill_impl = "flashmla_kv"
        backend.dsa_kv_cache_store_fp8 = True
        backend.use_mha = False
        backend.flashmla_kv_num_q_heads = 4
        backend.real_page_size = 64
        backend.kv_cache_dim = 656
        backend.dsa_index_topk = 3
        backend.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=lambda layer_id: producer_cache,
            set_mla_kv_buffer=lambda *args: produced.append(args),
        )
        layer = SimpleNamespace(
            layer_id=8,
            tp_q_head_num=2,
            head_dim=4,
            v_head_dim=2,
            scaling=0.75,
            is_cross_attention=False,
        )
        batch = SimpleNamespace(
            batch_size=2,
            forward_mode=ForwardMode.TARGET_VERIFY if extend else ForwardMode.DECODE,
            out_cache_loc=torch.arange(rows, dtype=torch.int32) + 7,
            seq_lens=torch.tensor([5, 7], dtype=torch.int32),
        )
        tile_metadata = torch.tensor([[11]], dtype=torch.int32)
        splits = torch.ones(rows, dtype=torch.int32)

        def flashmla(**operands):
            self.assertEqual(operands["q"].shape, (rows, 1, 4, 4))
            torch.testing.assert_close(operands["q"][:, 0, :2, :2], q_nope)
            torch.testing.assert_close(operands["q"][:, 0, :2, 2:], q_rope)
            self.assertEqual(torch.count_nonzero(operands["q"][:, :, 2:]).item(), 0)
            self.assertEqual(operands["k_cache"].data_ptr(), fetched_cache.data_ptr())
            self.assertEqual(operands["k_cache"].shape, (1, 64, 1, 656))
            torch.testing.assert_close(operands["indices"], (selected + 20)[:, None])
            torch.testing.assert_close(operands["cache_seqlens"], lengths)
            self.assertIs(operands["tile_scheduler_metadata"], tile_metadata)
            self.assertIs(operands["num_splits"], splits)
            self.assertEqual(operands["softmax_scale"], layer.scaling)
            return operands["q"][..., :2] + 10, None

        # Keep both native entrypoints, _compute_flashmla_metadata and the
        # unbound _forward_flashmla_kv callback real. Only CUDA operations and
        # external storage ownership are replaced with deterministic CPU seams.
        with (
            patch("torch.cuda.current_stream", return_value=stream),
            patch.object(
                dsa,
                "concat_mla_absorb_q_general",
                side_effect=lambda q, rope: torch.cat((q, rope), dim=-1),
            ),
            patch(
                "sgl_kernel.flash_mla.get_mla_metadata",
                return_value=(tile_metadata, splits),
            ) as schedule,
            patch(
                "sgl_kernel.flash_mla.flash_mla_with_kvcache", side_effect=flashmla
            ) as kernel,
        ):
            entry = (
                dsa.DeepseekSparseAttnBackend.forward_extend
                if extend
                else dsa.DeepseekSparseAttnBackend.forward_decode
            )
            output = entry(
                backend,
                q_nope if split_query else torch.cat((q_nope, q_rope), dim=-1),
                torch.ones((rows, 1, 2)),
                torch.ones((rows, 1, 2)),
                layer,
                batch,
                q_rope=q_rope if split_query else None,
                topk_indices=selected,
            )
        torch.testing.assert_close(output, q_nope[:, None] + 10)
        self.assertEqual(
            (len(fetches), kernel.call_count, schedule.call_count), (1, 1, 1)
        )
        self.assertEqual(schedule.call_args.kwargs["num_heads_q"], 4)
        self.assertEqual(schedule.call_args.kwargs["num_q_tokens_per_head_k"], 4)
        self.assertEqual(tuple(owner.slot for owner in fetches[0]["owners"]), (3, 5))
        torch.testing.assert_close(
            fetches[0]["selections"], selected.reshape(2, query_count, 3)
        )
        torch.testing.assert_close(fetches[0]["committed"], committed)
        expected_producers = (
            torch.tensor([[40, 41], [42, 43]]) if extend else torch.tensor([[47], [48]])
        )
        torch.testing.assert_close(fetches[0]["producer_locs"], expected_producers)
        torch.testing.assert_close(
            fetches[0]["query_limits"],
            (
                committed[:, None] + torch.arange(1, query_count + 1)[None]
                if extend
                else batch.seq_lens[:, None]
            ),
        )
        self.assertIs(fetches[0]["producer_cache"], producer_cache)
        self.assertEqual(consumed_layers, [8] if extend else [])
        self.assertEqual(released_native, [(0, stream, stream)])
        self.assertEqual(released_fetch, [stream])
        self.assertEqual(len(produced), 1)
        self.assertIs(produced[0][0], layer)
        self.assertIs(produced[0][1], batch.out_cache_loc)
        self.assertIsNone(metadata.flashmla_metadata)
        self.assertFalse(torch.cuda.is_initialized())

    def test_forward_extend_calls_unbound_native_callback_once(self):
        self._run(extend=True)

    def test_forward_decode_calls_unbound_native_callback_once(self):
        for split_query in (True, False):
            with self.subTest(split_query=split_query):
                self._run(extend=False, split_query=split_query)


class TestSM80BF16HiSparseDispatch(unittest.TestCase):
    """Check native ordinary BF16 dispatch with CPU producer/kernel seams.

    These tests qualify dispatch, physical addressing and prefill exclusion;
    CUDA transfers and native attention arithmetic require hardware checks.
    """

    def _run(self, *, prefill: bool, start_layer: int, split_query: bool) -> None:
        pool = HiSparseDSATokenToKVPool.__new__(HiSparseDSATokenToKVPool)
        pool.dtype = pool.store_dtype = torch.bfloat16
        pool.size = 32
        pool.page_size = 1
        pool.start_layer = start_layer
        pool.layer_transfer_counter = None
        pool.kv_buffer = [torch.zeros((33, 1, 576), dtype=torch.bfloat16)]
        pool.full_to_hisparse_device_index_mapping = torch.arange(33)
        pool.full_to_hisparse_device_index_mapping[-1] = -1
        pool.full_to_hisparse_device_index_mapping[3] = 17
        pool.full_to_hisparse_device_index_mapping[7] = 24
        locs = torch.tensor([3, 7], dtype=torch.int32)
        selected = torch.tensor([[0, 2, -1], [1, 3, -1]], dtype=torch.int32)
        physical = torch.tensor([[17, 9, -1], [24, 12, -1]], dtype=torch.int32)
        query = torch.ones((2, 2, 512), dtype=torch.bfloat16)
        rope = query[..., :64] * 2
        keys = query[:, :1] * torch.tensor([3, 7])[:, None, None]
        key_rope = keys[..., :64] * 2
        backend = dsa.DeepseekSparseAttnBackend.__new__(dsa.DeepseekSparseAttnBackend)
        backend.dsa_decode_impl = "triton-sm80"
        backend.dsa_prefill_impl = "tilelang"
        backend.use_mha = backend.use_fused_topk = False
        backend.token_to_kv_pool = pool
        backend.forward_metadata = SimpleNamespace(
            page_table_1=torch.tensor([[3, 9, 0], [7, 12, 0]], dtype=torch.int32),
            dsa_extend_seq_lens_list=[1, 1],
            cu_seqlens_q=torch.arange(3, dtype=torch.int32),
        )
        backend.get_topk_transform_method = lambda mode: dsa.TopkTransformMethod.PAGED
        co = SimpleNamespace(swap_in_selected_pages=None)
        backend.hisparse_coordinator = co
        layer = SimpleNamespace(
            layer_id=start_layer,
            tp_q_head_num=2,
            head_dim=576,
            v_head_dim=512,
            scaling=0.125,
            is_cross_attention=False,
        )
        batch = SimpleNamespace(
            batch_size=2,
            forward_mode=ForwardMode.EXTEND if prefill else ForwardMode.DECODE,
            out_cache_loc=locs,
            seq_lens=torch.tensor([3, 4], dtype=torch.int32),
            req_pool_indices=torch.tensor([2, 5], dtype=torch.int32),
        )

        def write(buffer, physical_locs, nope, tail):
            torch.testing.assert_close(physical_locs, torch.tensor([17, 24]))
            buffer[physical_locs.long()] = torch.cat((nope, tail), dim=-1)

        def materialize(requests, lengths, logical_selection, layer_id):
            self.assertIs(requests, batch.req_pool_indices)
            self.assertIs(lengths, batch.seq_lens)
            self.assertIs(logical_selection, selected)
            self.assertEqual(layer_id, start_layer)
            # Producers must already be written through the native remapping.
            torch.testing.assert_close(pool.kv_buffer[0][17, 0, :512], keys[0, 0])
            torch.testing.assert_close(pool.kv_buffer[0][24, 0, :512], keys[1, 0])
            return physical

        def decode(**operands):
            self.assertIs(operands["kv"], pool.kv_buffer[0])
            torch.testing.assert_close(operands["indices"], physical[:, None])
            torch.testing.assert_close(operands["q_nope"], query)
            torch.testing.assert_close(operands["q_rope"], rope)
            return query.unsqueeze(0)

        def prefill_attention(**operands):
            self.assertIs(operands["kv_cache"], pool.kv_buffer[0])
            torch.testing.assert_close(operands["page_table_1"], physical)
            return query

        from sglang.kernels.ops.attention.dsa import sparse_mla_decode_sm80 as sm80

        with (
            patch.object(
                memory_pool,
                "get_parallel",
                return_value=SimpleNamespace(attn_dcp_size=1),
            ),
            patch.object(
                memory_pool.MLATokenToKVPool, "_write_mla_kv_buffer", side_effect=write
            ),
            patch.object(sm80, "_require_sm80"),
            patch.object(sm80, "sparse_mla_decode_sm80", side_effect=decode) as kernel,
            patch.object(
                co, "swap_in_selected_pages", side_effect=materialize
            ) as fetch,
            patch.object(backend, "_forward_tilelang", side_effect=prefill_attention),
            patch.object(
                dsa,
                "concat_mla_absorb_q_general",
                side_effect=lambda main, tail: torch.cat((main, tail), dim=-1),
            ),
            patch.object(
                dsa,
                "transform_index_page_table_prefill",
                return_value=torch.tensor([[3, 9, -1], [7, 12, -1]], dtype=torch.int32),
            ),
            patch.object(
                dsa,
                "transform_index_page_table_decode",
                side_effect=AssertionError(
                    "HiSparse selections must use the materialization owner"
                ),
            ),
        ):
            entry = backend.forward_extend if prefill else backend.forward_decode
            output = entry(
                query if split_query else torch.cat((query, rope), dim=-1),
                keys,
                keys,
                layer,
                batch,
                q_rope=rope if split_query else None,
                k_rope=key_rope,
                topk_indices=selected,
            )
        torch.testing.assert_close(output, query)
        self.assertEqual(fetch.call_count, 0 if prefill else 1)
        self.assertEqual(kernel.call_count, 0 if prefill else 1)
        self.assertEqual(pool.kv_buffer[0][locs.long()].count_nonzero().item(), 0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_decode_uses_materialized_physical_rows_after_remapped_writes(self) -> None:
        for start_layer in (26, 52):
            for split_query in (False, True):
                with self.subTest(start_layer=start_layer, split_query=split_query):
                    self._run(
                        prefill=False, start_layer=start_layer, split_query=split_query
                    )

    def test_ordinary_prefill_reads_remapped_rows_without_selected_fetch(self) -> None:
        for start_layer in (26, 52):
            with self.subTest(start_layer=start_layer):
                self._run(prefill=True, start_layer=start_layer, split_query=True)

    def test_speculative_modes_fail_before_producing_or_fetching(self) -> None:
        backend = dsa.DeepseekSparseAttnBackend.__new__(dsa.DeepseekSparseAttnBackend)
        backend.dsa_decode_impl = "triton-sm80"
        backend.dsa_prefill_impl = "tilelang"
        query = torch.zeros((1, 2, 512), dtype=torch.bfloat16)
        for mode in (ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND_V2):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(NotImplementedError, "ordinary decode"):
                    backend.forward_extend(
                        query, query, query, None, SimpleNamespace(forward_mode=mode)
                    )


if __name__ == "__main__":
    unittest.main()
