"""Focused resident SM80 prefill scoring and current caller regressions."""

from __future__ import annotations

import ast
import contextlib
import math
import struct
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.kernels.ops.attention.dsa import torch_mqa_logits_sm80 as kernel
from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
from sglang.srt.layers.attention.dsa.ragged_mqa_logits_backend import (
    DSARaggedMQALogitsBackend,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=35, stage="base-b", runner_config="1-gpu-small")


def _decode(code: int) -> float:
    if code & 127 == 127:
        return math.nan
    exponent, fraction = divmod(code & 127, 8)
    value = (
        math.ldexp(fraction, -9)
        if exponent == 0
        else math.ldexp(8 + fraction, exponent - 10)
    )
    return math.copysign(value, -1 if code & 128 else 1)


def _bf16(value: float) -> float:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", rounded))[0]


def _fixture(rows: int = 4, heads: int = 4, keys: int = 65) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(42)
    q = (torch.randint(-8, 9, (rows, heads, 128), generator=generator).float() / 8).to(
        torch.float8_e4m3fn
    )
    k = (torch.randint(-8, 9, (keys, 128), generator=generator).float() / 8).to(
        torch.float8_e4m3fn
    )
    scales = torch.arange(1, keys + 1).float() / 4
    gates = torch.randint(-4, 5, (rows, heads), generator=generator).float() / 8
    starts = torch.tensor(([0, 1, keys, 5] * rows)[:rows], dtype=torch.int32)
    ends = torch.tensor(([keys, keys - 1, keys, 2] * rows)[:rows], dtype=torch.int32)
    return q, k, scales, gates, starts, ends


def _scalar_reference(inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    q, k, scales, gates, starts, ends = inputs
    raw_q = q.view(torch.uint8).tolist()
    raw_k = k.view(torch.uint8).tolist()
    output = torch.full((q.shape[0], k.shape[0]), -torch.inf, dtype=torch.float64)
    for row in range(q.shape[0]):
        for key in range(max(0, int(starts[row])), min(k.shape[0], int(ends[row]))):
            terms = []
            for head in range(q.shape[1]):
                dot = math.fsum(
                    _decode(x) * _decode(y)
                    for x, y in zip(raw_q[row][head], raw_k[key])
                )
                terms.append(max(0.0, _bf16(dot)) * float(gates[row, head]))
            output[row, key] = math.fsum(terms) * float(scales[key])
    return output


def _tensor_reference(inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    q, k, scales, gates, starts, ends = inputs
    dots = (q.double() @ k.double().T).to(torch.bfloat16).double().relu()
    output = (dots * gates.double()[..., None]).sum(1) * scales.double()
    positions = torch.arange(k.shape[0])[None, :]
    return output.masked_fill(
        (positions < starts[:, None]) | (positions >= ends[:, None]), -torch.inf
    )


def _indexer_methods() -> dict[str, object]:
    # Execute the actual maintained methods without importing a model/runtime.
    path = (
        Path(__file__).parents[5]
        / "python/sglang/srt/layers/attention/dsa/dsa_indexer.py"
    )
    tree = ast.parse(path.read_text())
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Indexer"
    )
    names = {
        "_get_cuda_ragged_mqa_logits",
        "_get_topk_ragged",
        "_should_skip_logits_computation",
        "forward_cuda",
    }
    methods = [
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "torch": torch,
        "TYPE_CHECKING": False,
        "_is_hip": False,
        "_is_fp8_fnuz": False,
    }
    ast_module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *methods,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(ast_module), str(path), "exec"), namespace)  # noqa: S102 - Test-only execution of maintained methods without model setup.
    return namespace


class TestSM80RaggedCPU(CustomTestCase):
    def test_independent_exact_policy_and_actual_cpu_candidate(self) -> None:
        inputs = _fixture()
        scalar, tensor = _scalar_reference(inputs), _tensor_reference(inputs)
        self.assertTrue(torch.equal(scalar, tensor))
        # The dyadic inputs/products and bounded integer sums are exact in FP64;
        # this verifies Python/operator logic on CPU, independently of GPU claims.
        raw = torch.empty_like(scalar)
        with (
            patch.object(kernel, "_require_sm80"),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
        ):
            output = kernel.torch_ragged_mqa_logits(
                *inputs, _verification_fp64=True, _verification_precast=raw
            )
            ordinary = kernel.torch_ragged_mqa_logits(*inputs)
        self.assertTrue(torch.equal(raw, scalar))
        self.assertTrue(torch.equal(output, scalar.float()))
        self.assertTrue(torch.equal(ordinary, output))

    def test_dot_bf16_midpoint_and_empty_results(self) -> None:
        q = torch.zeros(1, 1, 128)
        k = torch.zeros(1, 128)
        q[..., 0], k[..., 0] = 1, 1
        q[..., 1], k[..., 1] = 1 / 16, 1 / 16
        inputs = (
            q.to(torch.float8_e4m3fn),
            k.to(torch.float8_e4m3fn),
            torch.ones(1),
            torch.ones(1, 1),
            torch.zeros(1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
        )
        with (
            patch.object(kernel, "_require_sm80"),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
        ):
            output = kernel.torch_ragged_mqa_logits(*inputs, _verification_fp64=True)
            self.assertEqual(float(output[0, 0]), 1.0)
            for rows, keys in ((0, 1), (1, 0)):
                empty = _fixture(rows=rows, heads=1, keys=keys)
                self.assertEqual(
                    kernel.torch_ragged_mqa_logits(*empty).shape, (rows, keys)
                )

    def test_guards_and_tile_budget(self) -> None:
        inputs = _fixture(rows=1, heads=1, keys=1)
        with self.assertRaises(ValueError):
            kernel.torch_ragged_mqa_logits(*inputs)
        with patch.object(kernel, "_require_sm80"):
            for index, bad in (
                (0, inputs[0].float()),
                (1, torch.zeros(1, 127).to(torch.float8_e4m3fn)),
                (2, torch.ones(2)),
                (3, torch.ones(1, 2)),
                (4, torch.zeros(2, dtype=torch.int32)),
                (4, torch.zeros(1)),
                (0, torch.zeros(1, 1, 256).to(torch.float8_e4m3fn)[..., ::2]),
            ):
                changed = list(inputs)
                changed[index] = bad
                with self.assertRaises(ValueError):
                    kernel.torch_ragged_mqa_logits(*changed)
            with self.assertRaises(ValueError):
                kernel.torch_ragged_mqa_logits(
                    *inputs,
                    _verification_precast=torch.empty(1, 1, dtype=torch.float64),
                )
        for heads in (1, 4, 17, 32):
            row_tile, key_tile = kernel._tile_shape(8192, heads, 8192)
            forecast = (
                row_tile * heads * (kernel._SCORE_BYTES * key_tile + 128 * 8 + 8)
                + key_tile * 128 * 8
            )
            self.assertLessEqual(forecast, kernel._TEMP_BYTES)

        # Default selection forwards to the existing native scorer with the
        # existing head-padding decision; it never imports the SM80 scorer.
        namespace = _indexer_methods()
        native = Mock(return_value=torch.tensor([[123.0]]))
        namespace["deep_gemm"] = SimpleNamespace(fp8_mqa_logits=native)
        owner = SimpleNamespace(
            ragged_mqa_logits_backend=DSARaggedMQALogitsBackend.NATIVE,
            _pad_heads_for_deep_gemm=lambda q, weights: (q, weights, 1),
        )
        q, k, scales, weights, starts, ends = inputs
        output = namespace["_get_cuda_ragged_mqa_logits"](
            owner, q, (k, scales), weights, starts, ends
        )
        self.assertEqual(float(output[0, 0]), 123.0)
        native.assert_called_once()

    def test_selection_admission_namespace_and_early_mode_guard(self) -> None:
        from sglang.srt.layers.attention.dsa import (
            ragged_mqa_logits_backend as selection,
        )
        from sglang.srt.runtime_context import _build_config_bags
        from sglang.srt.server_args import ServerArgs

        args = ServerArgs(
            model_path="dummy", dsa_ragged_mqa_logits_backend="torch-sm80"
        )
        self.assertEqual(
            _build_config_bags(args)["exec"].kernel.dsa_ragged_mqa_logits_backend,
            "torch-sm80",
        )
        for capability in ((8, 0), (9, 0), (10, 0)):
            with (
                patch.object(selection, "is_cuda", return_value=True),
                patch.object(selection, "is_hip", return_value=False),
                patch.object(
                    selection, "get_device_capability", return_value=capability
                ),
            ):
                self.assertEqual(
                    selection.DSARaggedMQALogitsBackend.resolve("auto"),
                    DSARaggedMQALogitsBackend.NATIVE,
                )
                if capability == (8, 0):
                    self.assertTrue(
                        selection.DSARaggedMQALogitsBackend.resolve(
                            "torch-sm80"
                        ).is_torch_sm80()
                    )
                else:
                    with self.assertRaises(ValueError):
                        selection.DSARaggedMQALogitsBackend.resolve("torch-sm80")
        selected = DSARaggedMQALogitsBackend.TORCH_SM80
        config = {
            "hisparse": False,
            "speculative": False,
            "context_parallel": False,
            "paged_backend": "triton-sm80",
        }
        selected.validate_prefill_config(**config)
        for name in ("hisparse", "speculative", "context_parallel", "paged_backend"):
            bad = dict(config)
            bad[name] = "deepgemm" if name == "paged_backend" else True
            with self.assertRaises(ValueError):
                selected.validate_prefill_config(**bad)
        method = _indexer_methods()["forward_cuda"]
        owner = SimpleNamespace(ragged_mqa_logits_backend=selected)
        mode = SimpleNamespace(
            is_extend_without_speculative=lambda: False, is_decode_or_idle=lambda: False
        )
        for cp in (None, object()):
            batch = SimpleNamespace(attn_cp_metadata=cp, forward_mode=mode)
            with self.assertRaises(NotImplementedError):
                method(owner, None, None, None, batch, 0)


class TestSM80RaggedGPU(CustomTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            8,
            0,
        ):
            raise unittest.SkipTest("Requires actual SM80 CUDA execution")

    def test_exact_candidate_precast_and_admitted_views(self) -> None:
        for heads in (1, 4, 17):
            inputs = _fixture(heads=heads)
            expected = _scalar_reference(inputs)
            device_inputs = tuple(tensor.cuda() for tensor in inputs)
            raw = torch.empty_like(expected, device="cuda")
            controlled = kernel.torch_ragged_mqa_logits(
                *device_inputs, _verification_fp64=True, _verification_precast=raw
            )
            production = kernel.torch_ragged_mqa_logits(*device_inputs)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(raw.cpu(), expected))
            self.assertTrue(torch.equal(controlled.cpu(), expected.float()))
            self.assertTrue(torch.equal(production, controlled))
            # Sliced rows/heads preserve represented values and unit dimension strides.
            q, k, scales, gates, starts, ends = device_inputs
            changed = kernel.torch_ragged_mqa_logits(
                q[::2, ::2],
                k[::2],
                scales[::2],
                gates[::2, ::2],
                starts[::2],
                ends[::2],
            )
            changed_expected = _tensor_reference(
                tuple(
                    tensor.cpu()
                    for tensor in (
                        q[::2, ::2],
                        k[::2],
                        scales[::2],
                        gates[::2, ::2],
                        starts[::2],
                        ends[::2],
                    )
                )
            )
            self.assertTrue(torch.equal(changed.cpu(), changed_expected.float()))

    def test_current_direct_and_chunked_caller_above_select_all(self) -> None:
        from sglang.kernels.ops.attention.dsa.index_buf_accessor import GetKAndS

        namespace = _indexer_methods()
        rows, heads, keys = 3, 4, 2049
        q = torch.zeros(rows, heads, 128, device="cuda").to(torch.float8_e4m3fn)
        q.view(torch.uint8)[..., 0] = 0x38
        k = torch.zeros(keys, 128, device="cuda").to(torch.float8_e4m3fn)
        k.view(torch.uint8)[:, 0] = 0x38
        scales = torch.arange(1, keys + 1, device="cuda").float() / 2048
        gates = torch.ones(rows, heads, 1, device="cuda")
        starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
        ends = torch.tensor([2047, 2048, 2049], dtype=torch.int32, device="cuda")
        pages = (keys + 63) // 64
        packed = torch.zeros(pages, 64 * 132, dtype=torch.uint8, device="cuda")
        key_pages = torch.zeros(pages, 64, 128, dtype=torch.uint8, device="cuda")
        scale_pages = torch.zeros(pages, 64, dtype=torch.float32, device="cuda")
        key_pages.view(-1, 128)[:keys].copy_(k.view(torch.uint8))
        scale_pages.view(-1)[:keys].copy_(scales)
        packed[:, : 64 * 128].copy_(key_pages.flatten(1))
        packed[:, 64 * 128 :].copy_(scale_pages.view(torch.uint8))
        page_table = torch.arange(pages, dtype=torch.int32, device="cuda").view(1, -1)
        pool = SimpleNamespace(page_size=64, index_head_dim=128)
        pool.get_index_k_scale_buffer = (
            lambda layer_id, lengths, table, count, maximum: GetKAndS.execute(
                pool=pool,
                buf=packed,
                page_indices=table,
                seq_len_tensor=lengths,
                seq_len_sum=count,
                max_seq_len=maximum,
            )
        )
        namespace["get_token_to_kv_pool"] = lambda: pool
        mode = SimpleNamespace(is_extend_without_speculative=lambda: True)
        batch = SimpleNamespace(
            forward_mode=mode,
            seq_lens_cpu=torch.tensor([2049]),
            extend_seq_lens_cpu=torch.tensor([3]),
        )
        metadata = SimpleNamespace(
            get_page_table_64=lambda: page_table,
            get_indexer_kvcache_range=lambda: (starts, ends),
            get_indexer_seq_len_cpu=lambda: torch.tensor([keys]),
            get_indexer_seq_len=lambda: torch.tensor(
                [keys], dtype=torch.int32, device="cuda"
            ),
            get_seqlens_expanded=lambda: ends,
            get_token_to_batch_idx=lambda: [0, 0, 0],
            attn_metadata=SimpleNamespace(topk_indices_offset=starts),
        )
        metadata.topk_transform = lambda logits, topk, ks, **kwargs: (
            DSATopKBackend.TORCH.topk_func(
                logits,
                ends[: logits.shape[0]] - ks
                if "ke_offset" not in kwargs
                else kwargs["ke_offset"],
                topk,
                row_starts=ks,
            )
        )
        results = []
        for chunk in (False, True):
            owner = SimpleNamespace(
                index_topk=2048,
                ragged_mqa_logits_backend=DSARaggedMQALogitsBackend.TORCH_SM80,
                _with_real_sm_count=contextlib.nullcontext,
                _mask_init_and_local_tokens=lambda *args: None,
                _should_chunk_mqa_logits=lambda *args, chunk=chunk: (chunk, keys * 4),
            )
            owner._get_cuda_ragged_mqa_logits = MethodType(
                namespace["_get_cuda_ragged_mqa_logits"], owner
            )
            result = namespace["_get_topk_ragged"](
                owner, False, batch, 0, q, gates, metadata
            )
            torch.cuda.synchronize()
            results.append(result)
        for result in results:
            self.assertEqual(set(result[0].cpu().tolist()), {-1, *range(2047)})
            self.assertEqual(set(result[1].cpu().tolist()), set(range(2048)))
            self.assertEqual(set(result[2].cpu().tolist()), set(range(1, 2049)))
        self.assertTrue(torch.equal(results[0], results[1]))
        owner = SimpleNamespace(index_topk=2048)
        for length, skip in ((2048, True), (2049, False)):
            batch.seq_lens_cpu = torch.tensor([length])
            self.assertEqual(
                namespace["_should_skip_logits_computation"](owner, batch), skip
            )


if __name__ == "__main__":
    unittest.main()
