"""Focused sparse-MLA checks; exact scopes and weighted diagnostics are separate.

Zero-score cases have exact probabilities of one before the BF16 probability
cast. Fraction means independently qualify addressing and split/combine, with
FP64 final division uncertainty checked at the BF16 boundary. Nonzero-score
cases additionally compare the actual rounded-probability policy with a Decimal
reference; an empirical residual does not qualify libdevice's exp error bound.
The launch scale is represented as FP32 before upcasting. Controlled partial
states are widened internal intermediates; probability BF16 and result BF16
remain declared casts, with an auxiliary FP64 pre-result sink.
"""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from unittest.mock import patch

import torch
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=35, stage="base-b", runner_config="1-gpu-small")


@dataclass(frozen=True)
class _Inputs:
    nope: torch.Tensor
    rope: torch.Tensor
    pool: torch.Tensor
    indices: torch.Tensor
    scale: float = 0.125

    def on(self, device: str) -> _Inputs:
        return _Inputs(
            self.nope.to(device),
            self.rope.to(device),
            self.pool.to(device),
            self.indices.to(device),
            self.scale,
        )


def _fixture(tokens: int, heads: int, topk: int, *, weighted: bool = False) -> _Inputs:
    pool = torch.zeros(max(1, tokens * topk), 1, 576, dtype=torch.bfloat16)
    positions = torch.arange(pool.shape[0])
    pool[:, 0, 0] = ((positions % 7).float() - 3) / 8
    pool[:, 0, 1] = (positions % 3).float() / 4
    pool[:, 0, 7] = positions.float() / 8
    pool[:, 0, 511] = (positions % 4).float() / 8
    pool[:, 0, 512] = ((positions % 5).float() - 2) / 4
    indices = (
        torch.arange(tokens * topk, dtype=torch.int32).view(tokens, 1, topk).flip(-1)
    )
    nope = torch.zeros(tokens, heads, 512, dtype=torch.bfloat16)
    rope = torch.zeros(tokens, heads, 64, dtype=torch.bfloat16)
    if weighted:
        nope[..., 0] = 0.5
        rope[..., 0] = 2
    return _Inputs(nope, rope, pool, indices.contiguous())


def _logical_reference(inputs: _Inputs) -> torch.Tensor:
    out = torch.zeros(1, *inputs.nope.shape, dtype=torch.float64)
    pool = inputs.pool[:, 0].double()
    represented_scale = float(torch.tensor(inputs.scale, dtype=torch.float32))
    for row in range(inputs.nope.shape[0]):
        indices = inputs.indices[row, 0].long()
        selected = indices[(indices >= 0) & (indices < pool.shape[0])]
        if not selected.numel():
            continue
        keys = pool[selected]
        scores = (
            inputs.nope[row].double() @ keys[:, :512].T
            + inputs.rope[row].double() @ keys[:, 512:].T
        ) * represented_scale
        out[0, row] = scores.softmax(-1) @ keys[:, :512]
    return out


def _fraction_means(inputs: _Inputs) -> torch.Tensor:
    # This reference is only admitted when every query score is exactly zero.
    assert not bool(inputs.nope.count_nonzero()) and not bool(
        inputs.rope.count_nonzero()
    )
    result = torch.zeros(1, *inputs.nope.shape, dtype=torch.float64)
    rows = inputs.pool[:, 0].tolist()
    for token, indices in enumerate(inputs.indices[:, 0].tolist()):
        valid = [position for position in indices if 0 <= position < len(rows)]
        if not valid:
            continue
        for dimension in range(512):
            total = sum(
                (Fraction.from_float(rows[position][dimension]) for position in valid),
                Fraction(),
            )
            result[0, token, :, dimension] = float(total / len(valid))
    return result


def _decimal_policy_reference(
    inputs: _Inputs, splits: int, split_length: int
) -> torch.Tensor:
    # Scalar physical row lookup, independent of tensor gather and candidate.
    # Only active value dimensions need arithmetic; all others are exactly zero.
    result = torch.zeros(1, *inputs.nope.shape, dtype=torch.float64)
    rows = inputs.pool[:, 0].tolist()
    active = [d for d in range(512) if any(row[d] != 0 for row in rows)]
    with localcontext() as context:
        context.prec = 80
        scale = Decimal.from_float(
            float(torch.tensor(inputs.scale, dtype=torch.float32))
        )
        for token in range(inputs.nope.shape[0]):
            for head in range(inputs.nope.shape[1]):
                query = (
                    inputs.nope[token, head].tolist()
                    + inputs.rope[token, head].tolist()
                )
                active_query = [
                    (d, Decimal.from_float(value))
                    for d, value in enumerate(query)
                    if value
                ]
                partials: list[tuple[Decimal | None, Decimal, dict[int, Decimal]]] = []
                for split in range(splits):
                    maximum: Decimal | None = None
                    denominator = Decimal(0)
                    accumulated = {d: Decimal(0) for d in active}
                    start = split * split_length
                    end = min(start + split_length, inputs.indices.shape[-1])
                    for chunk in range(start, end, 64):
                        valid = [
                            int(index)
                            for index in inputs.indices[
                                token, 0, chunk : min(chunk + 64, end)
                            ]
                            if 0 <= int(index) < len(rows)
                        ]
                        if not valid:
                            continue
                        scores = [
                            sum(
                                (
                                    value * Decimal.from_float(rows[index][d])
                                    for d, value in active_query
                                ),
                                Decimal(),
                            )
                            * scale
                            for index in valid
                        ]
                        new_max = (
                            max(scores)
                            if maximum is None
                            else max(maximum, max(scores))
                        )
                        rescale = (
                            Decimal(0) if maximum is None else (maximum - new_max).exp()
                        )
                        probabilities = [(score - new_max).exp() for score in scores]
                        # The declared policy casts UNNORMALIZED probabilities,
                        # while the denominator still uses unrounded FP64 values.
                        rounded = [
                            Decimal.from_float(
                                float(torch.tensor(float(p), dtype=torch.bfloat16))
                            )
                            for p in probabilities
                        ]
                        denominator = denominator * rescale + sum(
                            probabilities, Decimal()
                        )
                        accumulated = {
                            d: accumulated[d] * rescale
                            + sum(
                                (
                                    p * Decimal.from_float(rows[index][d])
                                    for p, index in zip(rounded, valid, strict=True)
                                ),
                                Decimal(),
                            )
                            for d in active
                        }
                        maximum = new_max
                    partials.append((maximum, denominator, accumulated))
                finite_maxima = [m for m, _, _ in partials if m is not None]
                if not finite_maxima:
                    continue
                maximum = max(finite_maxima)
                denominator = Decimal(0)
                accumulated = {d: Decimal(0) for d in active}
                for local_max, local_sum, local_acc in partials:
                    if local_max is None:
                        continue
                    rescale = (local_max - maximum).exp()
                    denominator += local_sum * rescale
                    for d in active:
                        accumulated[d] += local_acc[d] * rescale
                for d in active:
                    result[0, token, head, d] = float(accumulated[d] / denominator)
    return result


def _kernel(
    inputs: _Inputs, *, fp64: bool = False, precast: torch.Tensor | None = None
) -> torch.Tensor:
    from sglang.kernels.ops.attention.dsa.sparse_mla_decode_sm80 import (
        sparse_mla_decode_sm80,
    )

    return sparse_mla_decode_sm80(
        inputs.nope,
        inputs.rope,
        inputs.pool,
        inputs.indices,
        inputs.scale,
        _verification_fp64=fp64,
        _verification_precast=precast,
    )


class TestSM80SparseMLAReferences(CustomTestCase):
    """Qualify bounded independent references without GPU computation."""

    def test_exact_zero_scores_and_independent_physical_means(self) -> None:
        inputs = _fixture(2, 1, 64)
        expected = _fraction_means(inputs)
        self.assertTrue(torch.equal(_logical_reference(inputs), expected))
        self.assertTrue(torch.equal(_decimal_policy_reference(inputs, 1, 64), expected))
        self.assertEqual(float(expected[0, 0, 0, 7]), 63 / 16)
        self.assertEqual(float(expected[0, 1, 0, 7]), 191 / 16)
        inputs.indices[0, 0, :32] = -1
        inputs.indices[1, 0] = inputs.pool.shape[0] + 1
        expected = _fraction_means(inputs)
        self.assertTrue(torch.equal(_decimal_policy_reference(inputs, 2, 64), expected))
        self.assertFalse(bool(expected[0, 1].count_nonzero()))

    def test_split_boundaries_cover_each_selected_row_once(self) -> None:
        from sglang.kernels.ops.attention.dsa.sparse_mla_decode_sm80 import (
            _split_k_config,
        )

        for topk in (1, 63, 64, 65, 129, 257):
            splits, length = _split_k_config(1, 17, topk, 108)
            self.assertEqual(splits & (splits - 1), 0)
            self.assertEqual(length % 64, 0)
            coverage = [
                i
                for split in range(splits)
                for i in range(split * length, min((split + 1) * length, topk))
            ]
            self.assertEqual(coverage, list(range(topk)))
        self.assertEqual(_split_k_config(1, 17, 129, 108), (4, 64))
        self.assertEqual(_split_k_config(1, 1, 129, 108), (4, 64))

    def test_probability_rounding_is_distinct_from_logical_attention(self) -> None:
        inputs = _fixture(1, 1, 65, weighted=True)
        logical = _logical_reference(inputs)
        policy = _decimal_policy_reference(inputs, 2, 64)
        self.assertGreater(float((logical - policy).abs().max()), 0)
        self.assertTrue(bool(torch.isfinite(policy).all()))
        # Input Python precision cannot recover information lost in the launch's
        # FP32 scalar representation.
        extra_python_bits = _Inputs(
            inputs.nope,
            inputs.rope,
            inputs.pool,
            inputs.indices,
            inputs.scale + 2.0**-30,
        )
        self.assertTrue(
            torch.equal(policy, _decimal_policy_reference(extra_python_bits, 2, 64))
        )
        self.assertTrue(torch.equal(logical, _logical_reference(extra_python_bits)))
        print(
            f"CPU policy witness: rounded-PV vs logical FP64 max_abs={float((logical - policy).abs().max()):.9g}; policies are distinct"
        )

    def test_resident_decode_configuration_guards(self) -> None:
        from sglang.kernels.ops.attention.dsa import sparse_mla_decode_sm80 as module

        with patch.object(module, "_require_sm80"):
            module.validate_sm80_sparse_mla_config(
                device=torch.device("cuda"),
                prefill_backend="fa3",
                kv_dtype=torch.bfloat16,
                hisparse=False,
                speculative=False,
            )
            for prefill, dtype, hisparse, speculative in (
                ("triton-sm80", torch.bfloat16, False, False),
                ("fa3", torch.float8_e4m3fn, False, False),
                ("fa3", torch.bfloat16, True, False),
                ("fa3", torch.bfloat16, False, True),
            ):
                with self.assertRaises(ValueError):
                    module.validate_sm80_sparse_mla_config(
                        device=torch.device("cuda"),
                        prefill_backend=prefill,
                        kv_dtype=dtype,
                        hisparse=hisparse,
                        speculative=speculative,
                    )
        with self.assertRaisesRegex(ValueError, "SM80"):
            module.validate_sm80_sparse_mla_config(
                device=torch.device("cpu"),
                prefill_backend="fa3",
                kv_dtype=torch.bfloat16,
                hisparse=False,
                speculative=False,
            )


class TestSM80SparseMLADecode(CustomTestCase):
    """Verify actual sparse-MLA kernels and declared BF16 cast boundaries."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            8,
            0,
        ):
            raise unittest.SkipTest("SM80 GPU required")

    def test_exact_attention_and_shared_fp64_split_combine(self) -> None:
        for tokens, heads, topk in (
            (1, 1, 1),
            (2, 17, 63),
            (1, 16, 64),
            (2, 1, 65),
            (1, 17, 129),
            (1, 32, 129),
        ):
            with self.subTest(tokens=tokens, heads=heads, topk=topk):
                cpu = _fixture(tokens, heads, topk)
                if topk == 129:
                    cpu.indices[0, 0, :64] = -1
                expected = _fraction_means(cpu)
                device = cpu.on("cuda")
                raw = torch.empty(expected.shape, dtype=torch.float64, device="cuda")
                controlled = _kernel(device, fp64=True, precast=raw)
                self.assertTrue(torch.equal(raw.cpu(), expected))
                self.assertTrue(torch.equal(controlled.cpu(), raw.cpu().bfloat16()))
                self.assertTrue(torch.equal(_kernel(device).cpu(), expected.bfloat16()))
                # Exact integer sums + one final correctly-rounded division;
                # the cast interval must remain strictly inside the BF16 cell.
                rounded = expected.bfloat16()
                below = torch.nextafter(
                    rounded, torch.full_like(rounded, -math.inf)
                ).double()
                above = torch.nextafter(
                    rounded, torch.full_like(rounded, math.inf)
                ).double()
                lower_mid = (below + rounded.double()) / 2
                upper_mid = (above + rounded.double()) / 2
                uncertainty = expected.abs() * (2.0**-52)
                nonzero = expected != 0
                self.assertTrue(
                    bool(
                        (
                            (expected - uncertainty > lower_mid)
                            & (expected + uncertainty < upper_mid)
                        )[nonzero].all()
                    )
                )
                print(
                    f"exact zero-score shared-FP64 T={tokens} H={heads} K={topk}: addressing/split/PV-cast/output PASS"
                )

    def test_nontrivial_weights_preserve_rounded_probability_policy(self) -> None:
        from sglang.kernels.ops.attention.dsa.sparse_mla_decode_sm80 import (
            _split_k_config,
        )

        cpu = _fixture(1, 1, 65, weighted=True)
        sm_count = torch.cuda.get_device_properties(0).multi_processor_count
        splits, length = _split_k_config(1, 1, 65, sm_count)
        policy = _decimal_policy_reference(cpu, splits, length)
        logical = _logical_reference(cpu)
        raw = torch.empty(policy.shape, dtype=torch.float64, device="cuda")
        controlled = _kernel(cpu.on("cuda"), fp64=True, precast=raw)
        actual_raw = raw.cpu()
        self.assertTrue(torch.equal(controlled.cpu(), actual_raw.bfloat16()))
        self.assertTrue(torch.equal(controlled.cpu(), policy.bfloat16()))
        residual = float((actual_raw - policy).abs().max())
        self.assertLess(residual, 1e-11)
        ordinary = _kernel(cpu.on("cuda")).cpu()
        print(
            f"weighted diagnostics: actual-FP64 vs Decimal-policy max_abs={residual:.9g}; logical-vs-policy={float((logical - policy).abs().max()):.9g}; ordinary strict BF16 parity={torch.equal(ordinary, controlled.cpu())}; exp arithmetic bound unqualified"
        )

    def test_all_invalid_and_empty_selections(self) -> None:
        cpu = _fixture(2, 17, 65)
        cpu.indices[0].fill_(-1)
        cpu.indices[1].fill_(cpu.pool.shape[0])
        self.assertFalse(bool(_kernel(cpu.on("cuda")).count_nonzero()))
        self.assertFalse(bool(_kernel(cpu.on("cuda"), fp64=True).count_nonzero()))
        empty = _fixture(1, 1, 0)
        self.assertEqual(tuple(_kernel(empty.on("cuda")).shape), (1, 1, 1, 512))
        self.assertFalse(bool(_kernel(empty.on("cuda")).count_nonzero()))
        empty_sink = torch.full(
            (1, 1, 1, 512), math.nan, dtype=torch.float64, device="cuda"
        )
        empty_controlled = _kernel(empty.on("cuda"), fp64=True, precast=empty_sink)
        self.assertFalse(bool(empty_controlled.count_nonzero()))
        self.assertFalse(bool(empty_sink.count_nonzero()))

    def test_strided_query_views_and_graph_replay(self) -> None:
        from sglang.kernels.ops.attention.dsa.sparse_mla_decode_sm80 import (
            _split_k_config,
        )

        cpu = _fixture(2, 17, 65, weighted=True)
        cpu.rope[..., 0] = torch.arange(17).to(torch.bfloat16) / 8
        device = cpu.on("cuda")
        q_storage = torch.zeros(2, 34, 576, device="cuda", dtype=torch.bfloat16)
        q_storage[:, ::2, :512] = device.nope
        q_storage[:, ::2, 512:] = device.rope
        pool_storage = torch.zeros(
            device.pool.shape[0], 1, 1152, device="cuda", dtype=torch.bfloat16
        )
        pool_storage[..., :576] = device.pool
        index_storage = torch.full((2, 1, 130), -1, device="cuda", dtype=torch.int32)
        index_storage[..., :65] = device.indices
        views = _Inputs(
            q_storage[:, ::2, :512],
            q_storage[:, ::2, 512:],
            pool_storage[..., :576],
            index_storage[..., :65],
        )
        splits, length = _split_k_config(
            2, 17, 65, torch.cuda.get_device_properties(0).multi_processor_count
        )
        policy = _decimal_policy_reference(cpu, splits, length)
        raw = torch.empty(policy.shape, device="cuda", dtype=torch.float64)
        controlled = _kernel(views, fp64=True, precast=raw)
        self.assertTrue(torch.equal(controlled.cpu(), policy.bfloat16()))
        for _ in range(3):
            _kernel(views)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = _kernel(views)
        for _ in range(2):
            views.indices.copy_(views.indices.flip(0))
            views.pool[:, 0, 7].mul_(-1)
            q_storage[:, :, 512].neg_()
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(output, _kernel(views)))


if __name__ == "__main__":
    unittest.main()
