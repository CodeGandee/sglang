"""CPU contracts for native token-major pipeline graph input buffers.

The genuine allocator, sharing pool, registry and runner prepare/load owners run
on CPU tensors. Attention metadata planning is an explicit CPU carrier; CUDA
graphs, model arithmetic and distributed pipeline transport are not executed.
"""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention import dsa_backend as attention
from sglang.srt.model_executor import input_buffers as shared
from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    CudaGraphBufferRegistry,
    build_decode_registry,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.runner_utils.buffers import DecodeInputBuffers
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _buffers(
    bs: int = 1,
    tokens: int = 8,
    *,
    pp: int = 4,
    hc_width: int | None = None,
    residual_blocks: int | None = None,
    topk: int | None = None,
    aux: list[str] | None = None,
) -> DecodeInputBuffers:
    return DecodeInputBuffers.create(
        device=torch.device("cpu"),
        max_bs=bs,
        max_num_token=tokens,
        hidden_size=6144,
        next_token_logits_buffer=torch.zeros(tokens, 5),
        dtype=torch.bfloat16,
        dp_size=1,
        pp_size=pp,
        is_encoder_decoder=False,
        require_mlp_tp_gather=False,
        seq_len_fill_value=6,
        encoder_len_fill_value=0,
        num_tokens_per_req=8,
        cache_loc_dtype=torch.int64,
        enable_mamba_track=True,
        hc_hidden_size=hc_width,
        pp_proxy_topk_size=topk,
        pp_proxy_residual_num_blocks=residual_blocks,
        pp_proxy_aux_keys=aux,
    )


def _registry(buffers: DecodeInputBuffers) -> CudaGraphBufferRegistry:
    return build_decode_registry(
        device=torch.device("cpu"),
        max_bs=buffers.req_pool_indices.numel(),
        max_num_token=buffers.input_ids.numel(),
        seq_len_fill_value=6,
        cache_loc_dtype=torch.int64,
        enable_mamba_track=True,
        source=buffers,
    )


def _runner(buffers: DecodeInputBuffers) -> DecodeCudaGraphRunner:
    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    backend = attention.DeepseekSparseAttnBackend.__new__(
        attention.DeepseekSparseAttnBackend
    )
    backend.device = torch.device("cpu")
    runner.buffers = buffers
    runner.buffer_registry = _registry(buffers)
    runner.captured_req_width = 8
    runner.pp_size = 4
    runner.dp_size = 1
    runner.require_gathered_buffer = False
    runner.enable_prefill_cp = False
    runner.require_mlp_tp_gather = False
    runner.require_attn_tp_gather = False
    runner.capture_hidden_mode = CaptureHiddenMode.FULL
    runner.capture_forward_mode = ForwardMode.TARGET_VERIFY
    runner.attn_backend = backend
    runner.model_runner = cast(
        ModelRunner,
        SimpleNamespace(
            server_args=SimpleNamespace(enable_lora=False),
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            attn_backend=backend,
            is_draft_worker=False,
            hisparse_coordinator=None,
        ),
    )
    runner.ragged_verify_mode = False
    runner.enable_pdmux = False
    runner.enable_two_batch_overlap = False
    runner.is_encoder_decoder = False
    runner.seq_len_fill_value = 6
    runner.capture_bs = [1, buffers.req_pool_indices.numel()]
    runner.deepep_adapter = Mock(replay=Mock())
    return runner


class TestDecodePPTokenBuffers(unittest.TestCase):
    """Keep request rows separate from token rows through actual CPU owners."""

    def setUp(self) -> None:
        topology = get_parallel().override(moe_ep_size=1)
        topology.__enter__()
        self.addCleanup(topology.__exit__, None, None, None)
        empty_pool: dict[shared._PoolKey, torch.Tensor] = {}
        pool = patch.object(shared, "_forward_input_buffer_pool", empty_pool)
        pool.start()
        self.addCleanup(pool.stop)

    def test_request_bucket_allocates_all_verification_rows(self) -> None:
        for bs, tokens in ((1, 8), (2, 16)):
            buffers = _buffers(bs, tokens)
            assert buffers.pp_proxy_tensors is not None
            assert buffers.mamba_track_indices is not None
            self.assertEqual(
                buffers.pp_proxy_tensors["hidden_states"].shape, (tokens, 6144)
            )
            self.assertEqual(buffers.pp_proxy_tensors["residual"].shape, (tokens, 6144))
            self.assertEqual(buffers.req_pool_indices.shape, (bs,))
            self.assertEqual(buffers.seq_lens.shape, (bs,))
            self.assertEqual(buffers.seq_lens_cpu.shape, (bs,))
            self.assertEqual(buffers.mamba_track_indices.shape, (bs,))
            self.assertEqual(buffers.input_ids.shape, (tokens,))
            self.assertEqual(buffers.out_cache_loc.shape, (tokens,))

    def test_ordinary_one_token_requests_preserve_shapes(self) -> None:
        buffers = _buffers(3, 3)
        assert buffers.pp_proxy_tensors is not None
        self.assertEqual(buffers.pp_proxy_tensors["hidden_states"].shape, (3, 6144))
        self.assertEqual(buffers.pp_proxy_tensors["residual"].shape, (3, 6144))
        self.assertEqual(buffers.req_pool_indices.shape, (3,))

    def test_special_residual_topk_and_mhc_geometry_stays_distinct(self) -> None:
        blocks = _buffers(residual_blocks=4, topk=2048)
        assert blocks.pp_proxy_tensors is not None
        self.assertEqual(blocks.pp_proxy_tensors["residual"].shape, (8, 4, 6144))
        self.assertEqual(blocks.pp_proxy_tensors["topk_indices"].shape, (8, 2048))
        self.assertEqual(blocks.pp_proxy_tensors["topk_indices"].dtype, torch.int32)
        mhc = _buffers(hc_width=8192)
        assert mhc.pp_proxy_tensors is not None
        self.assertEqual(mhc.pp_proxy_tensors["hidden_states"].shape, (8, 8192))
        self.assertNotIn("residual", mhc.pp_proxy_tensors)
        self.assertIsNone(_buffers(pp=1).pp_proxy_tensors)

    def test_sharing_never_adopts_request_sized_pipeline_backing(self) -> None:
        old = shared.share_input_buffer(
            "pp_proxy_tensors.hidden_states", torch.zeros(1, 6144, dtype=torch.bfloat16)
        )
        first, second = _buffers(), _buffers()
        assert first.pp_proxy_tensors is not None
        assert second.pp_proxy_tensors is not None
        first.share_buffers()
        second.share_buffers()
        for name in ("hidden_states", "residual"):
            left, right = first.pp_proxy_tensors[name], second.pp_proxy_tensors[name]
            self.assertEqual(left.shape, (8, 6144))
            self.assertEqual(left.data_ptr(), right.data_ptr())
            self.assertNotEqual(left.data_ptr(), old.data_ptr())

    def test_registry_adopts_original_full_token_backing(self) -> None:
        buffers = _buffers(topk=2048)
        assert buffers.pp_proxy_tensors is not None
        buffers.share_buffers()
        registry = _registry(buffers)
        for name, value in buffers.pp_proxy_tensors.items():
            self.assertIs(registry.get_slot("pp_proxy_tensors." + name).buffer, value)
        self.assertIs(
            registry.get_slot("req_pool_indices").buffer, buffers.req_pool_indices
        )
        self.assertEqual(registry.get_slot("req_pool_indices").buffer.shape, (1,))

    def test_actual_capture_and_replay_load_keep_all_rows_and_aliases(self) -> None:
        for bs in (1, 2):
            buffers = _buffers(bs, bs * 8)
            assert buffers.pp_proxy_tensors is not None
            buffers.share_buffers()
            runner = _runner(buffers)
            forward, backend, proxy = runner.capture_prepare(bs)
            self.assertEqual(forward.batch_size, bs)
            self.assertEqual(forward.input_ids.shape, (bs * 8,))
            self.assertEqual(proxy.tensors["hidden_states"].shape, (bs * 8, 6144))
            self.assertEqual(proxy.tensors["residual"].shape, (bs * 8, 6144))
            for name, value in proxy.tensors.items():
                self.assertEqual(
                    value.data_ptr(), buffers.pp_proxy_tensors[name].data_ptr()
                )
            # A native latent key geometry must retain every verification row.
            key = proxy.tensors["hidden_states"][:, :64].contiguous()
            self.assertEqual(key.view(bs * 8, -1, 64).shape, (bs * 8, 1, 64))
            pointers = {name: value.data_ptr() for name, value in proxy.tensors.items()}
            source = replace(
                forward,
                input_ids=torch.arange(bs * 8),
                positions=torch.arange(6144, 6144 + bs * 8),
                req_pool_indices=torch.arange(bs),
                seq_lens=torch.full((bs,), 6144, dtype=torch.int64),
                seq_lens_cpu=torch.full((bs,), 6144, dtype=torch.int64),
                out_cache_loc=torch.arange(100, 100 + bs * 8),
            )
            with patch.object(backend, "init_forward_metadata_out_graph") as metadata:
                for step in range(3):
                    payload = PPProxyTensors(
                        {
                            "hidden_states": torch.full(
                                (bs * 8, 6144), float(step + 1), dtype=torch.bfloat16
                            ),
                            "residual": torch.full(
                                (bs * 8, 6144), -float(step + 1), dtype=torch.bfloat16
                            ),
                        }
                    )
                    payload.tensors["hidden_states"][0, 0] = -0.0
                    runner.load_batch(source, payload)
                    self.assertEqual(metadata.call_count, step + 1)
                    for name, value in proxy.tensors.items():
                        self.assertEqual(value.data_ptr(), pointers[name])
                        self.assertTrue(
                            torch.equal(
                                value.view(torch.uint16),
                                payload.tensors[name].view(torch.uint16),
                            )
                        )
                    self.assertEqual(runner.raw_bs, bs)
                    self.assertEqual(runner.raw_num_token, bs * 8)
                    self.assertEqual(runner._replay_graph_key.size, bs)

    def test_preplanned_replay_refreshes_pipeline_without_attention_replanning(
        self,
    ) -> None:
        for bs in (1, 2):
            buffers = _buffers(bs, bs * 8, residual_blocks=3, topk=7)
            assert buffers.pp_proxy_tensors is not None
            runner = _runner(buffers)
            forward, backend, proxy = runner.capture_prepare(bs)
            pointers = {name: bank.data_ptr() for name, bank in proxy.tensors.items()}
            request_pointers = {
                name: getattr(buffers, name).data_ptr()
                for name in ("req_pool_indices", "seq_lens", "seq_lens_cpu")
            }
            with patch.object(backend, "init_forward_metadata_out_graph") as metadata:
                # The actual DFlash preplanner has no pipeline inputs yet.
                runner.load_batch(forward)
                self.assertEqual(metadata.call_count, 1)
                forward.apply_deprecated_skip_attn_backend_init(True)
                self.assertFalse(forward.needs_forward_metadata_init())
                key = runner._replay_graph_key
                for step in range(3):
                    forward.input_ids = torch.full_like(forward.input_ids, step + 17)
                    forward.positions = torch.arange(bs * 8) + 4154 + step
                    incoming = PPProxyTensors(
                        {
                            name: torch.full_like(bank, step + 1)
                            for name, bank in proxy.tensors.items()
                        }
                    )
                    incoming.tensors["hidden_states"][0, 0] = -0.0
                    runner.load_batch(forward, incoming)
                    self.assertEqual(metadata.call_count, 1)
                    self.assertFalse(forward.needs_forward_metadata_init())
                    self.assertEqual(
                        (runner.raw_bs, runner.raw_num_token), (bs, bs * 8)
                    )
                    self.assertEqual(runner._replay_graph_key, key)
                    self.assertTrue(torch.equal(buffers.input_ids, forward.input_ids))
                    self.assertTrue(torch.equal(buffers.positions, forward.positions))
                    for name, bank in proxy.tensors.items():
                        self.assertEqual(bank.data_ptr(), pointers[name])
                        self.assertIs(
                            runner.buffer_registry.get_slot(
                                "pp_proxy_tensors." + name
                            ).buffer,
                            buffers.pp_proxy_tensors[name],
                        )
                        self.assertTrue(
                            torch.equal(
                                bank.view(torch.uint8),
                                incoming.tensors[name].view(torch.uint8),
                            )
                        )
                for name, pointer in request_pointers.items():
                    self.assertEqual(getattr(buffers, name).data_ptr(), pointer)
                    self.assertEqual(getattr(buffers, name).shape, (bs,))

    def test_preplanned_pipeline_copy_preserves_source_extent_and_absent_inputs(
        self,
    ) -> None:
        buffers = _buffers(2, 16)
        assert buffers.pp_proxy_tensors is not None
        runner = _runner(buffers)
        forward, backend, _ = runner.capture_prepare(1)
        for bank in buffers.pp_proxy_tensors.values():
            bank.fill_(-9)
        with patch.object(backend, "init_forward_metadata_out_graph") as metadata:
            runner.load_batch(forward)
            forward.mark_forward_metadata_ready()
            incoming = PPProxyTensors(
                {
                    name: torch.full_like(bank[:8], 5)
                    for name, bank in buffers.pp_proxy_tensors.items()
                }
            )
            runner.load_batch(forward, incoming)
            for bank in buffers.pp_proxy_tensors.values():
                self.assertTrue((bank[:8] == 5).all())
                self.assertTrue((bank[8:] == -9).all())
            before = {
                name: bank.clone() for name, bank in buffers.pp_proxy_tensors.items()
            }
            runner.load_batch(forward, None)
            for name, bank in buffers.pp_proxy_tensors.items():
                self.assertTrue(torch.equal(bank, before[name]))
            self.assertEqual(metadata.call_count, 1)

    def test_dflash_aux_pipeline_buffers_allocate_adopt_and_refresh(self) -> None:
        aux = ["dflash_aux_l5", "dflash_aux_l23"]
        buffers = _buffers(2, 16, aux=aux)
        assert buffers.pp_proxy_tensors is not None
        for name in aux:
            self.assertEqual(buffers.pp_proxy_tensors[name].shape, (16, 6144))
            self.assertEqual(buffers.pp_proxy_tensors[name].dtype, torch.bfloat16)
        self.assertIsNone(_buffers(pp=1, aux=aux).pp_proxy_tensors)
        buffers.share_buffers()
        registry = _registry(buffers)
        for name in aux:
            self.assertIs(
                registry.get_slot("pp_proxy_tensors." + name).buffer,
                buffers.pp_proxy_tensors[name],
            )
        runner = _runner(buffers)
        forward, backend, _ = runner.capture_prepare(1)
        for bank in buffers.pp_proxy_tensors.values():
            bank.fill_(-9)
        with patch.object(backend, "init_forward_metadata_out_graph") as metadata:
            runner.load_batch(forward)
            forward.mark_forward_metadata_ready()
            incoming = PPProxyTensors(
                {
                    name: torch.full_like(bank[:8], 5)
                    for name, bank in buffers.pp_proxy_tensors.items()
                }
            )
            runner.load_batch(forward, incoming)
            for name, bank in buffers.pp_proxy_tensors.items():
                self.assertTrue((bank[:8] == 5).all(), name)
                self.assertTrue((bank[8:] == -9).all(), name)
            self.assertEqual(metadata.call_count, 1)

    def test_resolve_pp_proxy_dflash_aux_keys_matches_stage_ownership(self) -> None:
        from sglang.srt.model_executor.model_runner_components import misc_utils

        glm = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"])
        )
        features = [5, 23, 41, 59]
        # GLM-5.3 runs 78 layers; a PP4 stage owns a half-open native range
        # and receives exactly the features captured at upstream layer outputs.
        for pp_rank, start_layer, expected in (
            (0, 0, None),
            (1, 20, ["dflash_aux_l5"]),
            (2, 40, ["dflash_aux_l5", "dflash_aux_l23"]),
            (3, 59, ["dflash_aux_l5", "dflash_aux_l23", "dflash_aux_l41"]),
        ):
            with self.subTest(pp_rank=pp_rank):
                self.assertEqual(
                    misc_utils.resolve_pp_proxy_dflash_aux_keys(
                        model_config=glm,
                        pp_size=4,
                        pp_rank=pp_rank,
                        start_layer=start_layer,
                        dflash_target_layer_ids=features,
                    ),
                    expected,
                )
        # A feature whose capture point opens the stage is local, not relayed.
        self.assertIsNone(
            misc_utils.resolve_pp_proxy_dflash_aux_keys(
                model_config=glm,
                pp_size=4,
                pp_rank=1,
                start_layer=20,
                dflash_target_layer_ids=[19],
            )
        )
        for pp_size, ids in ((1, features), (4, None), (4, [])):
            with self.subTest(pp_size=pp_size, ids=ids):
                self.assertIsNone(
                    misc_utils.resolve_pp_proxy_dflash_aux_keys(
                        model_config=glm,
                        pp_size=pp_size,
                        pp_rank=3,
                        start_layer=59,
                        dflash_target_layer_ids=ids,
                    )
                )
        with self.assertRaisesRegex(RuntimeError, "post-layer capture"):
            misc_utils.resolve_pp_proxy_dflash_aux_keys(
                model_config=SimpleNamespace(
                    hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"])
                ),
                pp_size=4,
                pp_rank=1,
                start_layer=20,
                dflash_target_layer_ids=features,
            )

    def test_dflash_aux_key_contract_matches_predkv_plugin(self) -> None:
        from predkv_extensions.dflash2 import capture as predkv_capture
        from predkv_extensions.dflash2 import pp_worker as predkv_pp_worker
        from sglang.srt.model_executor.model_runner_components import misc_utils

        glm = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"])
        )
        resolved = misc_utils.resolve_pp_proxy_dflash_aux_keys(
            model_config=glm,
            pp_size=4,
            pp_rank=3,
            start_layer=59,
            dflash_target_layer_ids=[5, 23, 41, 59],
        )
        assert resolved is not None
        self.assertEqual(
            resolved,
            [f"{predkv_capture.AUX_PREFIX}{i}" for i in (5, 23, 41)],
        )
        self.assertEqual(resolved, [predkv_pp_worker.aux_key(i) for i in (5, 23, 41)])

    def test_registry_refuses_oversized_pipeline_payload(self) -> None:
        buffers = _buffers()
        runner = _runner(buffers)
        forward, _, _ = runner.capture_prepare(1)
        payload = PPProxyTensors(
            {
                "hidden_states": torch.ones(16, 6144, dtype=torch.bfloat16),
                "residual": torch.ones(16, 6144, dtype=torch.bfloat16),
            }
        )
        with self.assertRaises(RuntimeError):
            runner.load_batch(forward, payload)

    def test_padded_replay_consumes_live_rows_and_keeps_captured_output(self) -> None:
        from sglang.srt.layers.attention.dsa import sm80_dflash_verify as verify
        from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
            BreakableCudaGraphBackend,
        )
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
            get_graph_forward_batch,
            graph_forward_batch_scope,
        )

        # Real native allocator/registry/capture/load/replay owners run on CPU.
        # Metadata, grouped lease consumption and graph execution are explicit
        # source-test leaves, not model arithmetic or GPU lifetime qualification.
        for raw_bs, bucket in ((1, 2), (2, 4)):
            with self.subTest(raw_bs=raw_bs, bucket=bucket):
                buffers = _buffers(bucket, bucket * 8)
                runner = _runner(buffers)
                runner.capture_bs = [bucket]
                # Hardware identity and recipe admission are explicit CPU
                # leaves; native mask policy and input construction stay real.
                runner.attn_backend.device = torch.device("cuda")
                with (
                    patch.object(
                        verify, "sm80_dflash_any_config_admitted", return_value=True
                    ),
                    patch.object(
                        torch.cuda, "get_device_capability", return_value=(8, 0)
                    ),
                ):
                    captured, native_backend, _ = runner.capture_prepare(bucket)
                    live_source, _, _ = runner.capture_prepare(raw_bs)
                self.assertIsNone(captured.spec_info.custom_mask)
                self.assertIsNone(live_source.spec_info.custom_mask)
                backend = cast(Any, native_backend)
                rows = raw_bs * 8
                live = replace(
                    live_source,
                    batch_size=raw_bs,
                    spec_info=replace(
                        live_source.spec_info,
                        draft_token=torch.zeros(rows, dtype=torch.int64),
                        positions=torch.arange(rows) + 5,
                    ),
                    input_ids=torch.arange(rows),
                    positions=torch.arange(rows) + 5,
                    req_pool_indices=torch.arange(raw_bs) + 1,
                    seq_lens=torch.full((raw_bs,), 5, dtype=torch.int64),
                    seq_lens_cpu=torch.full((raw_bs,), 5, dtype=torch.int64),
                    out_cache_loc=torch.arange(rows) + 32,
                )
                with patch.object(backend, "init_forward_metadata_out_graph"):
                    runner.load_batch(live)
                self.assertEqual((runner.raw_bs, runner.bs), (raw_bs, bucket))
                self.assertEqual(runner.raw_num_token, rows)
                metadata = attention.DSAMetadata(
                    page_size=64,
                    cache_seqlens_int32=torch.full((bucket,), 5),
                    max_seq_len_q=8,
                    max_seq_len_k=16,
                    cu_seqlens_q=torch.arange(bucket + 1) * 8,
                    cu_seqlens_k=torch.arange(bucket + 1) * 5,
                    page_table_1=torch.zeros(bucket * 8, 1, dtype=torch.int32),
                    real_page_table=torch.zeros(bucket * 8, 1, dtype=torch.int32),
                    dsa_cache_seqlens_int32=torch.full((bucket * 8,), 5),
                    dsa_cu_seqlens_q=torch.arange(bucket * 8 + 1),
                    dsa_cu_seqlens_k=torch.arange(bucket * 8 + 1) * 5,
                    dsa_extend_seq_lens_list=[8] * bucket,
                    dsa_seqlens_expanded=torch.full((bucket * 8,), 5),
                )
                cache = torch.zeros(256, 1, 576, dtype=torch.bfloat16)
                write = Mock()
                backend.token_to_kv_pool = SimpleNamespace(
                    dtype=torch.bfloat16,
                    set_mla_kv_buffer=write,
                    get_key_buffer=lambda _layer: cache,
                )
                backend.use_mha = backend.use_fused_topk = False
                backend.speculative_num_draft_tokens = 8
                backend.real_page_size = 64
                backend.forward_metadata = metadata
                owners = tuple(
                    SimpleNamespace(scratch=torch.arange(8) + 100 + row * 8)
                    for row in range(raw_bs)
                )
                transaction = SimpleNamespace(
                    owners=owners, block_size=8, committed=torch.full((raw_bs,), 5)
                )
                consume = Mock(
                    return_value=torch.full((rows, 2, 512), 3, dtype=torch.bfloat16)
                )
                backend.hisparse_coordinator = SimpleNamespace(
                    _group_attention=SimpleNamespace(record_bytes=1152),
                    _target_kv=SimpleNamespace(_round=transaction),
                    attend_group=consume,
                )
                layer = SimpleNamespace(
                    tp_q_head_num=2,
                    head_dim=576,
                    v_head_dim=512,
                    is_cross_attention=False,
                    layer_id=0,
                    scaling=1.0,
                )
                q = torch.ones(bucket * 8, 2, 576, dtype=torch.bfloat16)
                k = torch.ones(bucket * 8, 1, 512, dtype=torch.bfloat16)
                rope = torch.ones(bucket * 8, 1, 64, dtype=torch.bfloat16)
                selection = torch.zeros(bucket * 8, 2048, dtype=torch.int64)
                result: list[torch.Tensor] = []

                def eager_break() -> None:
                    self.assertIs(get_graph_forward_batch(), live)
                    result.append(
                        backend._forward_sm80_dflash_verify(
                            q, k, k, layer, captured, True, None, rope, selection
                        )
                    )

                graph = cast(
                    Any, BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend)
                )
                key = object()
                graph._graphs = {key: SimpleNamespace(replay=eager_break)}
                graph._aux_packers = {key: ()}
                output = object()
                graph._outputs = {key: output}
                with (
                    patch.object(
                        verify, "sm80_dflash_any_config_admitted", return_value=True
                    ),
                    patch.object(
                        verify,
                        "get_schedule",
                        return_value=SimpleNamespace(max_running_requests=raw_bs),
                    ),
                    graph_forward_batch_scope(live),
                ):
                    masked = replace(
                        live,
                        spec_info=replace(
                            live.spec_info,
                            custom_mask=torch.ones(rows, dtype=torch.bool),
                        ),
                    )
                    with self.assertRaises(NotImplementedError):
                        verify.require_sm80_dflash_verify(masked, q.shape[0])
                    write.assert_not_called()
                    consume.assert_not_called()
                    self.assertIs(graph.replay(key, live), output)
                    self.assertIs(get_graph_forward_batch(), live)
                    with self.assertRaisesRegex(RuntimeError, "foreign target"):
                        graph.replay(
                            key,
                            replace(live, req_pool_indices=live.req_pool_indices + 1),
                        )
                    transaction.owners = (*owners, owners[0])
                    with self.assertRaisesRegex(
                        RuntimeError, "actual grouped target transaction"
                    ):
                        graph.replay(key, live)
                    transaction.owners = owners
                self.assertIsNone(get_graph_forward_batch())
                self.assertEqual(captured.batch_size, bucket)
                self.assertEqual(captured.req_pool_indices.shape, (bucket,))
                self.assertEqual(result[0].shape, (bucket * 8, 2, 512))
                self.assertTrue((result[0][:rows] == 3).all())
                self.assertEqual(torch.count_nonzero(result[0][rows:]), 0)
                self.assertEqual(consume.call_count, 1)
                args = consume.call_args.args
                self.assertEqual(args[2].shape[0], rows)
                self.assertEqual(args[7].dsa_cache_seqlens_int32.shape, (rows,))
                self.assertEqual(args[8].shape, (raw_bs, 8, 2048))
                self.assertEqual(args[10].shape, (raw_bs, 8))
                write.assert_called_once()
                self.assertIs(write.call_args.args[1], live.out_cache_loc)
                self.assertEqual(write.call_args.args[2].shape[0], rows)
                self.assertEqual(write.call_args.args[3].shape[0], rows)
                self.assertEqual(metadata.dsa_cache_seqlens_int32.shape, (bucket * 8,))

    def test_replay_frame_restores_on_graph_and_auxiliary_failure(self) -> None:
        from sglang.srt.model_executor.runner_backend import (
            breakable_cuda_graph_backend as bcg,
        )
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
            get_graph_forward_batch,
        )

        forward, _, _ = _runner(_buffers()).capture_prepare(1)
        graph = cast(
            Any, bcg.BreakableCudaGraphBackend.__new__(bcg.BreakableCudaGraphBackend)
        )
        key = object()
        graph._graphs = {key: Mock(replay=Mock())}
        graph._aux_packers = {key: ()}
        graph._outputs = {key: object()}
        for owner in (graph._graphs[key].replay, None):
            failure = ValueError("replay failed")
            if owner is not None:
                owner.side_effect = failure
            with patch.object(bcg, "publish_aux_replay", side_effect=failure):
                with self.assertRaises(ValueError):
                    graph.replay(key, forward)
            self.assertIsNone(get_graph_forward_batch())
            graph._graphs[key].replay.side_effect = None


if __name__ == "__main__":
    unittest.main()
