"""Check actual Torch selection, metadata and index domains without GPUs.

Triton transform bodies execute through the CPU interpreter. The FlashMLA kernel
boundary records indices; this does not qualify FlashMLA numerical execution.
"""

import ast
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from triton.runtime.interpreter import InterpretedFunction

from sglang.kernels.ops.attention.dsa import transform_index as transforms
from sglang.srt.environ import envs
from sglang.srt.layers.attention import dsa_backend as dsa
from sglang.srt.layers.attention.dsa import utils
from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    DSATopKBackend,
    TopkTransformMethod,
)
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative import eagle_disaggregation as eagle
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
TOPK = 2048


def _actual_block(method, select, arguments, return_expression):
    source = Path(dsa.__file__)
    module = ast.parse(source.read_text())
    owner = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "DeepseekSparseAttnBackend"
    )
    definition = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == method
    )
    statements = [node for node in definition.body if select(node)]
    if len(statements) != 1:
        raise AssertionError("Native index-domain block changed")
    function = ast.parse(f"def probe({arguments}):\n    pass\n").body[0]
    function.body = [
        *statements,
        ast.Return(value=ast.parse(return_expression, mode="eval").body),
    ]
    compiled = ast.Module(
        body=[ast.parse("from __future__ import annotations").body[0], function],
        type_ignores=[],
    )
    namespace = dict(vars(dsa))
    exec(compile(ast.fix_missing_locations(compiled), str(source), "exec"), namespace)
    return namespace["probe"]


def _resolve(backend):
    def assignment(node):
        return isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Attribute) and target.attr == "use_fused_topk"
            for target in node.targets
        )

    return _actual_block(
        "__init__",
        assignment,
        "self, seed_dsa_topk_from_draft_extend=False",
        "self.use_fused_topk",
    )(backend)


def _runtime(stack, backend, *, pd="null", hisparse=False, dcp=False):
    stack.enter_context(
        patch.object(
            utils,
            "get_exec",
            return_value=SimpleNamespace(
                kernel=SimpleNamespace(dsa_topk_backend=backend.value)
            ),
            create=True,
        )
    )
    stack.enter_context(
        patch.object(
            utils, "get_disagg", return_value=SimpleNamespace(disaggregation_mode=pd)
        )
    )
    stack.enter_context(
        patch.object(
            utils, "get_memory", return_value=SimpleNamespace(enable_hisparse=hisparse)
        )
    )
    stack.enter_context(
        patch.object(
            utils, "get_parallel", return_value=SimpleNamespace(dcp_enabled=dcp)
        )
    )


def _interpreter(stack):
    for name in (
        "transform_index_page_table_prefill_kernel",
        "transform_index_page_table_decode_kernel",
    ):
        kernel = getattr(transforms, name)
        stack.enter_context(
            patch.object(transforms, name, InterpretedFunction(kernel.fn))
        )


class _CPUBackend(dsa.DeepseekSparseAttnBackend):
    def __init__(self, table, lengths, extend, *, hisparse=False, ragged=False):
        self.dsa_topk_backend = DSATopKBackend.TORCH
        _resolve(self)
        self.dsa_prefill_impl = "flashmla_sparse" if ragged else "flashmla_kv"
        self.dsa_decode_impl = "flashmla_kv"
        self.dsa_kv_cache_store_fp8 = True
        self.use_mha = False
        self.hisparse_coordinator = (
            SimpleNamespace(swap_in_selected_pages=Mock()) if hisparse else None
        )
        mapping = torch.arange(int(table.max()) + 2, dtype=torch.int32) * 2 + 1
        mapping[-1] = -1
        self.token_to_kv_pool = HiSparseDSATokenToKVPool.__new__(
            HiSparseDSATokenToKVPool
        )
        self.token_to_kv_pool.register_mapping(mapping)
        self.token_to_kv_pool.get_key_buffer = Mock(return_value=torch.zeros((1, 1, 2)))
        self._forward_flashmla_kv = Mock(
            side_effect=lambda **kwargs: kwargs["page_table_1"]
        )
        cu = [0]
        for count in extend:
            cu.append(cu[-1] + count)
        self.forward_metadata = SimpleNamespace(
            page_table_1=table,
            real_page_table=table,
            cache_seqlens_int32=lengths,
            dsa_seqlens_expanded=lengths,
            dsa_extend_seq_lens_list=extend,
            cu_seqlens_q=torch.tensor(cu, dtype=torch.int32),
            topk_indices_offset=None,
            paged_mqa_schedule_metadata=None,
            paged_mqa_ctx_lens_2d=None,
        )


def _scores_and_table():
    width = 2305
    columns = torch.arange(width)
    scores = torch.stack((columns, -columns, columns * 17 % 2311, columns)).to(
        torch.float32
    )
    backing = torch.empty((2, width * 2), dtype=torch.int32)
    table = backing[:, ::2]
    table[0] = torch.arange(width - 1, -1, -1) + 11
    table[1] = torch.arange(width) + width + 51
    return (
        scores,
        table,
        torch.tensor([3, 5, 2300, 0], dtype=torch.int32),
        torch.tensor([0, 0, 5, 5], dtype=torch.int32),
    )


def _scalar_select(scores, starts, lengths):
    rows = []
    for row in range(scores.shape[0]):
        begin, count = int(starts[row]), int(lengths[row])
        selected = sorted(
            range(begin, begin + count),
            key=lambda col: float(scores[row, col]),
            reverse=True,
        )[:TOPK]
        local = [col - begin for col in selected]
        rows.append(local + [-1] * (TOPK - len(local)))
    return torch.tensor(rows, dtype=torch.int32)


def _scalar_map(table, indices, request_rows):
    return torch.tensor(
        [
            [
                int(table[request, int(position)]) if int(position) >= 0 else -1
                for position in indices[row]
            ]
            for row, request in enumerate(request_rows)
        ],
        dtype=torch.int32,
    )


def _batch(mode):
    return SimpleNamespace(
        forward_mode=mode,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([5, 2300], dtype=torch.int32),
        out_cache_loc=torch.arange(5, dtype=torch.int32),
    )


class TestTorchTopKDomains(unittest.TestCase):
    def test_actual_constructor_resolves_backend_capability_without_env_mutation(self):
        for backend in DSATopKBackend:
            for requested in (False, True):
                with (
                    self.subTest(backend=backend, requested=requested),
                    ExitStack() as stack,
                ):
                    stack.enter_context(envs.SGLANG_DSA_FUSE_TOPK.override(requested))
                    _runtime(stack, backend)
                    owner = SimpleNamespace(dsa_topk_backend=backend)
                    self.assertEqual(
                        _resolve(owner), requested and backend != DSATopKBackend.TORCH
                    )
                    self.assertEqual(envs.SGLANG_DSA_FUSE_TOPK.get(), requested)

    def test_pd_constraints_and_torch_graph_keep_request_local_mapping(self):
        cases = (
            ("null", False, False, True),
            ("prefill", False, False, False),
            ("decode", False, False, True),
            ("decode", True, False, False),
            ("decode", False, True, False),
        )
        for selected in DSATopKBackend:
            for pd, hisparse, dcp, supported in cases:
                for requested in (False, True):
                    with (
                        self.subTest(
                            backend=selected,
                            pd=pd,
                            hisparse=hisparse,
                            dcp=dcp,
                            requested=requested,
                        ),
                        ExitStack() as stack,
                    ):
                        stack.enter_context(
                            envs.SGLANG_DSA_FUSE_TOPK.override(requested)
                        )
                        stack.enter_context(
                            patch.object(utils, "is_cuda", return_value=True)
                        )
                        _runtime(stack, selected, pd=pd, hisparse=hisparse, dcp=dcp)
                        self.assertEqual(
                            utils.should_use_dsa_fused_topk(True),
                            requested
                            and supported
                            and selected != DSATopKBackend.TORCH,
                        )
                        self.assertEqual(envs.SGLANG_DSA_FUSE_TOPK.get(), requested)

        def assignment(node):
            return isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Attribute)
                and target.attr == "dsa_drop_wide_page_table"
                for target in node.targets
            )

        owner = SimpleNamespace(
            dsa_topk_backend=DSATopKBackend.TORCH,
            use_fused_topk=False,
            real_page_size=64,
            hisparse_coordinator=None,
            speculative_num_draft_tokens=0,
            dsa_index_topk=TOPK,
        )
        with (
            patch.object(dsa, "is_cuda", return_value=True),
            patch.object(dsa, "_is_hip", False),
        ):
            self.assertFalse(
                _actual_block(
                    "init_cuda_graph_state",
                    assignment,
                    "self",
                    "self.dsa_drop_wide_page_table",
                )(owner)
            )

    def test_actual_paged_prefill_maps_once_and_preserves_padding_and_payload_values(
        self,
    ):
        scores, table, lengths, starts = _scores_and_table()
        self.assertFalse(table.is_contiguous())
        for hisparse in (False, True):
            with self.subTest(hisparse=hisparse), ExitStack() as stack:
                stack.enter_context(envs.SGLANG_DSA_FUSE_TOPK.override(True))
                _runtime(stack, DSATopKBackend.TORCH, hisparse=hisparse)
                _interpreter(stack)
                stack.enter_context(
                    patch.object(
                        dsa,
                        "concat_mla_absorb_q_general",
                        side_effect=lambda a, b: torch.cat((a, b), dim=-1),
                    )
                )
                backend = _CPUBackend(table, lengths, [2, 1], hisparse=hisparse)
                batch = _batch(ForwardMode.EXTEND)
                metadata = backend.get_indexer_metadata(0, batch)
                self.assertTrue(metadata.force_unfused_topk)
                raw = metadata.topk_transform(scores, TOPK, ks=starts)
                expected_raw = _scalar_select(scores, starts, lengths)
                self.assertTrue(torch.equal(raw, expected_raw))
                expected = torch.full((5, TOPK), -1, dtype=torch.int32)
                expected[:3] = _scalar_map(table, expected_raw[:3], [0, 0, 1])
                if hisparse:
                    expected = torch.where(expected >= 0, expected * 2 + 1, expected)
                layer = SimpleNamespace(
                    is_cross_attention=False,
                    layer_id=0,
                    tp_q_head_num=1,
                    v_head_dim=1,
                    head_dim=2,
                    scaling=1.0,
                )
                actual = backend.forward_extend(
                    torch.zeros((5, 1)),
                    None,
                    None,
                    layer,
                    batch,
                    save_kv_cache=False,
                    q_rope=torch.zeros((5, 1)),
                    topk_indices=raw,
                )
                self.assertTrue(torch.equal(actual, expected))
                backend._forward_flashmla_kv.assert_called_once()
                if hisparse:
                    backend.hisparse_coordinator.swap_in_selected_pages.assert_not_called()
                # A scalar cache-value reference makes the logical/physical
                # distinction numerically observable without a FlashMLA kernel.
                values = actual.clamp(min=0).to(torch.float64) * 7 + 3
                mean = torch.where(actual >= 0, values, 0).sum(1) / (actual >= 0).sum(
                    1
                ).clamp(min=1)
                for row in range(5):
                    physical = [
                        int(index) for index in expected[row] if int(index) >= 0
                    ]
                    reference = (
                        sum(index * 7 + 3 for index in physical) / len(physical)
                        if physical
                        else 0
                    )
                    self.assertEqual(float(mean[row]), reference)

    def test_short_k_only_actual_indexer_preserves_all_valid_entries_and_static_padding(
        self,
    ):
        _, table, _, _ = _scores_and_table()
        lengths = torch.tensor([3, 5, 0, 2], dtype=torch.int32)
        with ExitStack() as stack:
            stack.enter_context(envs.SGLANG_DSA_FUSE_TOPK.override(True))
            _runtime(stack, DSATopKBackend.TORCH, hisparse=True)
            backend = _CPUBackend(table, lengths, [2, 1], hisparse=True)
            metadata = backend.get_indexer_metadata(0, _batch(ForwardMode.EXTEND))
            owner = SimpleNamespace(
                use_dsa_indexer_fusion=False,
                index_topk=TOPK,
                _get_k_bf16=Mock(return_value=torch.zeros((5, 2))),
                _store_index_k_cache=Mock(),
            )
            output = torch.full((5, TOPK), -1, dtype=torch.int32)
            pointer = output.data_ptr()
            returned = Indexer._forward_cuda_k_only(
                owner,
                torch.zeros((5, 2)),
                torch.arange(5),
                _batch(ForwardMode.EXTEND),
                0,
                None,
                metadata,
                num_tokens=3,
                topk_result=output,
            )
            self.assertIsNone(returned)
            self.assertEqual(output.data_ptr(), pointer)
            for row, length in enumerate(lengths):
                self.assertEqual(
                    sorted(output[row][output[row] >= 0].tolist()),
                    list(range(int(length))),
                )
                self.assertEqual(int((output[row] == -1).sum()), TOPK - int(length))
            self.assertTrue(torch.all(output[4] == -1))
            owner._store_index_k_cache.assert_called_once()

    def test_ordinary_ragged_consumer_adds_nonzero_request_offset_once(self):
        scores, table, lengths, starts = _scores_and_table()
        with ExitStack() as stack:
            stack.enter_context(envs.SGLANG_DSA_FUSE_TOPK.override(True))
            _runtime(stack, DSATopKBackend.TORCH)
            backend = _CPUBackend(table, lengths, [2, 1], ragged=True)
            backend.forward_metadata.topk_indices_offset = torch.tensor(
                [0, 0, 5, 5], dtype=torch.int32
            )
            batch = _batch(ForwardMode.EXTEND)
            metadata = backend.get_indexer_metadata(0, batch)
            raw = metadata.topk_transform(scores, TOPK, ks=starts)
            expected = _scalar_select(scores, starts, lengths)
            expected = torch.where(expected >= 0, expected + starts[:, None], expected)

            def block(node):
                return (
                    isinstance(node, ast.If)
                    and ast.unparse(node.test) == "self.use_fused_topk"
                )

            consumer = _actual_block(
                "forward_extend",
                block,
                "self, topk_indices, q_nope, metadata, forward_batch, topk_transform_method",
                "topk_indices",
            )
            result = consumer(
                backend,
                raw,
                torch.zeros((4, 1)),
                backend.forward_metadata,
                batch,
                TopkTransformMethod.RAGGED,
            )
            self.assertTrue(torch.equal(result, expected))

    def test_hisparse_decode_and_idle_pass_logical_positions_to_coordinator(self):
        _, table, _, _ = _scores_and_table()
        lengths = torch.tensor([3, 2300], dtype=torch.int32)
        scores = torch.arange(table.shape[1], dtype=torch.float32).repeat(2, 1)
        with ExitStack() as stack:
            stack.enter_context(envs.SGLANG_DSA_FUSE_TOPK.override(True))
            _runtime(stack, DSATopKBackend.TORCH, hisparse=True)
            stack.enter_context(
                patch.object(
                    dsa,
                    "concat_mla_absorb_q_general",
                    side_effect=lambda a, b: torch.cat((a, b), dim=-1),
                )
            )
            backend = _CPUBackend(table, lengths, [1, 1], hisparse=True)
            batch = _batch(ForwardMode.DECODE)
            for mode in (ForwardMode.DECODE, ForwardMode.IDLE):
                self.assertTrue(
                    backend.get_indexer_metadata(0, _batch(mode)).force_unfused_topk
                )
            raw = backend.get_indexer_metadata(0, batch).topk_transform(scores, TOPK)
            expected = _scalar_select(
                scores, torch.zeros(2, dtype=torch.int32), lengths
            )
            self.assertTrue(torch.equal(raw, expected))
            working = torch.full_like(raw, -1)
            working[:, :3] = torch.tensor([71, 72, 73])
            backend.hisparse_coordinator.swap_in_selected_pages.return_value = working
            layer = SimpleNamespace(
                is_cross_attention=False,
                layer_id=0,
                tp_q_head_num=1,
                v_head_dim=1,
                head_dim=2,
                scaling=1.0,
            )
            result = backend.forward_decode(
                torch.zeros((2, 2)),
                None,
                None,
                layer,
                batch,
                save_kv_cache=False,
                topk_indices=raw,
            )
            arguments = (
                backend.hisparse_coordinator.swap_in_selected_pages.call_args.args
            )
            self.assertIs(arguments[2], raw)
            self.assertTrue(torch.equal(result, working))

    def test_pd_torch_seed_stays_raw_and_supported_fused_seed_remaps_once(self):
        table = torch.tensor([[19, 17, 13, 11], [79, 83, 81, 89]], dtype=torch.int32)
        seeds = torch.full((2, TOPK), -1, dtype=torch.int32)
        seeds[:, :2] = torch.tensor([2, 0])
        requests = [
            SimpleNamespace(
                output_topk_p=[1.0],
                output_topk_index=[1],
                hidden_states_tensor=torch.zeros(2),
                output_dsa_topk_indices=seed,
            )
            for seed in seeds
        ]
        batch = SimpleNamespace(
            reqs=requests,
            device="cpu",
            enable_overlap=False,
            req_pool_indices=torch.tensor([0, 1]),
            req_to_token_pool=SimpleNamespace(req_to_token=table),
            seq_lens=torch.tensor([4, 4]),
        )
        physical = _scalar_map(table, seeds, [0, 1])
        for backend in DSATopKBackend:
            with self.subTest(backend=backend), ExitStack() as stack:
                stack.enter_context(envs.SGLANG_DSA_FUSE_TOPK.override(True))
                _runtime(stack, backend, pd="decode")
                stack.enter_context(patch.object(utils, "is_cuda", return_value=True))
                stack.enter_context(
                    patch.object(
                        eagle,
                        "get_spec",
                        return_value=SimpleNamespace(
                            speculative_eagle_topk=1, enable_multi_layer_eagle=False
                        ),
                    )
                )
                _interpreter(stack)
                draft = eagle.build_eagle_disagg_draft_input(
                    batch, torch.tensor([11, 12]), None
                )
                if backend == DSATopKBackend.TORCH:
                    self.assertFalse(utils.should_use_dsa_fused_topk(True))
                    self.assertTrue(torch.equal(draft.dsa_topk_indices, seeds))
                    transformed = transforms.transform_index_page_table_decode(
                        page_table=table, topk_indices=draft.dsa_topk_indices
                    )
                    self.assertTrue(torch.equal(transformed, physical))
                else:
                    self.assertTrue(utils.should_use_dsa_fused_topk(True))
                    self.assertTrue(torch.equal(draft.dsa_topk_indices, physical))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
