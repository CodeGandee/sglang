"""Qualify a bounded nonuniform weighted SM80 attention policy.

Represented scores are exactly 0 or -1024. Since e > 5/2 and (5/2)^4 > 32,
exp(-1024) < 2^-1280, strictly below half the smallest FP64 subnormal.
Correctly rounded FP64 exponentials are therefore exactly 0 or 1. Scalar
witnesses exercise the actual candidate exponential helper separately. This
qualifies saturated weighted cases; general weighted exponential error remains
unqualified. Byte transfer and full-model semantics are outside this file.
For the finite fixture, ignoring the true tiny weights changes each active
result by less than K * 16 * 2^-1280; rational cast-cell checks retain this bound.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from fractions import Fraction

import torch
import triton
import triton.language as tl
from sglang.kernels.ops.attention.dsa.sparse_mla_decode_sm80 import (
    _attention_exp,
    _split_k_config,
    sparse_mla_decode_sm80,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-small")


@triton.jit
def _exp_probe(inputs, outputs, VERIFY_FP64: tl.constexpr):
    offsets = tl.arange(0, 4)
    values = tl.load(inputs + offsets)
    result = _attention_exp(values, VERIFY_FP64)
    tl.store(outputs + offsets, result)


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


def _fixture(topk: int, *, changed: bool = False) -> _Inputs:
    rows = 1 << max(1, (topk + 1).bit_length())
    pool = torch.zeros(rows, 1, 576, dtype=torch.bfloat16)
    pool[:, 0, 512] = -4096 if changed else -8192
    # Non-surviving values make erroneous inclusion observable in the result.
    pool[:, 0, 0] = -7
    pool[:, 0, 511] = 5
    query = 2 if changed else 1
    rope = torch.zeros(1, 32, 64, dtype=torch.bfloat16)
    rope[..., 0] = query
    nope = torch.zeros(1, 32, 512, dtype=torch.bfloat16)
    logical = torch.arange(topk, dtype=torch.int64)
    # Odd multipliers independently permute power-of-two physical capacities.
    physical = (logical * (19 if changed else 37) + (23 if changed else 11)) % rows
    indices = physical.to(torch.int32).view(1, 1, topk)
    if topk:
        survivor_positions = (0,) if topk == 1 else (0, topk - 1)
        for number, position in enumerate(survivor_positions):
            row = int(physical[position])
            pool[row, 0, 512] = 0
            sign = -1 if changed else 1
            pool[row, 0, 0] = sign * (3 + number * 2) / 8
            pool[row, 0, 7] = sign * (2 + number * 4) / 8
            pool[row, 0, 511] = sign * (1 + number * 2) / 16
    if topk > 4:
        indices[0, 0, 1] = -1
        indices[0, 0, 2] = rows
    return _Inputs(nope, rope, pool, indices)


def _exact_reference(inputs: _Inputs) -> torch.Tensor:
    """Use scalar represented operands and exact rational saturated weights."""
    if inputs.nope.device.type != "cpu":
        raise ValueError("Independent reference requires retained CPU operands")
    represented_scale = Fraction.from_float(
        float(torch.tensor(inputs.scale, dtype=torch.float32))
    )
    pool = inputs.pool[:, 0].tolist()
    result = torch.zeros(1, *inputs.nope.shape, dtype=torch.float64)
    for token in range(inputs.nope.shape[0]):
        selected = [
            int(row)
            for row in inputs.indices[token, 0].tolist()
            if 0 <= int(row) < len(pool)
        ]
        for head in range(inputs.nope.shape[1]):
            query = (
                inputs.nope[token, head].tolist() + inputs.rope[token, head].tolist()
            )
            active_query = [
                (dimension, Fraction.from_float(value))
                for dimension, value in enumerate(query)
                if value
            ]
            scores = [
                sum(
                    (
                        value * Fraction.from_float(pool[row][dimension])
                        for dimension, value in active_query
                    ),
                    Fraction(),
                )
                * represented_scale
                for row in selected
            ]
            if any(score not in (0, -1024) for score in scores):
                raise ValueError("Reference admits only exact scores 0/-1024")
            if not scores:
                continue
            maximum = max(scores)
            survivors = [
                row
                for row, score in zip(selected, scores, strict=True)
                if score == maximum
            ]
            if len(survivors) & (len(survivors) - 1):
                raise ValueError("Reference requires a power-of-two surviving count")
            for dimension in (0, 7, 511):
                exact = sum(
                    (Fraction.from_float(pool[row][dimension]) for row in survivors),
                    Fraction(),
                ) / len(survivors)
                result[0, token, head, dimension] = float(exact)
            if any(
                value != 0
                for row in survivors
                for dimension, value in enumerate(pool[row][:512])
                if dimension not in (0, 7, 511)
            ):
                raise ValueError(
                    "Fixture contains an undeclared active value dimension"
                )
    return result


class TestSM80HiSparseAttentionReferences(unittest.TestCase):
    """Check exact arithmetic and address declarations without CUDA.

    Notes
    -----
    These checks qualify the independent fixture/reference, not native kernels.
    """

    def test_underflow_bound_and_independent_weighted_reference(self) -> None:
        # A finite Taylor lower bound, rather than an empirical exp comparison.
        self.assertGreater(
            Fraction(1) + 1 + Fraction(1, 2) + Fraction(1, 6), Fraction(5, 2)
        )
        self.assertGreater(Fraction(5, 2) ** 4, 2**5)
        self.assertGreater(1280, 1075)
        for changed in (False, True):
            expected = _exact_reference(_fixture(65, changed=changed))
            sign = -1 if changed else 1
            self.assertTrue(bool((expected[0, :, :, 0] == sign * 0.5).all()))
            self.assertTrue(bool((expected[0, :, :, 7] == sign * 0.5).all()))
            self.assertTrue(bool((expected[0, :, :, 511] == sign * 0.125).all()))
            self.assertTrue(torch.equal(expected, expected.bfloat16().double()))

    def test_invalid_masks_duplicates_and_unsupported_scores(self) -> None:
        inputs = _fixture(65)
        inputs.indices.fill_(-1)
        self.assertFalse(bool(_exact_reference(inputs).count_nonzero()))
        inputs.indices.fill_(inputs.pool.shape[0])
        self.assertFalse(bool(_exact_reference(inputs).count_nonzero()))
        duplicate = _fixture(2)
        duplicate.indices[0, 0, 1] = duplicate.indices[0, 0, 0]
        self.assertEqual(float(_exact_reference(duplicate)[0, 0, 0, 0]), 3 / 8)
        unsupported = _fixture(2)
        unsupported.pool[int(unsupported.indices[0, 0, 0]), 0, 512] = -8
        with self.assertRaisesRegex(ValueError, "exact scores"):
            _exact_reference(unsupported)
        non_dyadic = _fixture(3)
        non_dyadic.pool[non_dyadic.indices[0, 0].long(), 0, 512] = 0
        with self.assertRaisesRegex(ValueError, "power-of-two"):
            _exact_reference(non_dyadic)

    def test_true_softmax_underflow_error_stays_inside_bf16_cells(self) -> None:
        for changed in (False, True):
            expected = _exact_reference(_fixture(2049, changed=changed))
            rounded = expected.bfloat16()
            below = torch.nextafter(rounded, torch.full_like(rounded, -float("inf")))
            above = torch.nextafter(rounded, torch.full_like(rounded, float("inf")))
            # All values have magnitude at most 7, surviving count >= 1. The
            # omitted-weight perturbation is < K*(7+7)*2^-1280; 16 is a strict
            # conservative bound. Zero inactive dimensions stay exactly zero.
            error = Fraction(2049 * 16, 2**1280)
            for dimension in (0, 7, 511):
                value = Fraction.from_float(float(expected[0, 0, 0, dimension]))
                lower = (
                    Fraction.from_float(float(below[0, 0, 0, dimension])) + value
                ) / 2
                upper = (
                    Fraction.from_float(float(above[0, 0, 0, dimension])) + value
                ) / 2
                self.assertGreater(value - error, lower)
                self.assertLess(value + error, upper)


class TestSM80HiSparseSaturatedAttention(unittest.TestCase):
    """Exercise actual weighted native attention with proved cast boundaries.

    Notes
    -----
    General weighted exponentials and whole-model semantics remain unqualified.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Require the explicitly selected SM80 CUDA device before execution."""
        super().setUpClass()
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            8,
            0,
        ):
            raise unittest.SkipTest("SM80 GPU required")

    def test_actual_exponential_special_values(self) -> None:
        for fp64 in (False, True):
            dtype = torch.float64 if fp64 else torch.float32
            values = torch.tensor(
                [0, -1024, -float("inf"), 0], dtype=dtype, device="cuda"
            )
            observed = torch.empty_like(values)
            _exp_probe[(1,)](values, observed, VERIFY_FP64=fp64)
            self.assertTrue(
                torch.equal(observed.cpu(), torch.tensor([1, 0, 0, 1], dtype=dtype))
            )

    def test_actual_weighted_splits_permutations_and_changed_data(self) -> None:
        self.test_actual_exponential_special_values()
        for topk in (1, 63, 64, 65, 129, 2047, 2048, 2049):
            for changed in (False, True):
                with self.subTest(topk=topk, changed=changed):
                    cpu = _fixture(topk, changed=changed)
                    expected = _exact_reference(cpu)
                    inputs = cpu.on("cuda")
                    raw = torch.empty_like(expected, device="cuda")
                    controlled = sparse_mla_decode_sm80(
                        inputs.nope,
                        inputs.rope,
                        inputs.pool,
                        inputs.indices,
                        inputs.scale,
                        _verification_fp64=True,
                        _verification_precast=raw,
                    )
                    ordinary = sparse_mla_decode_sm80(
                        inputs.nope,
                        inputs.rope,
                        inputs.pool,
                        inputs.indices,
                        inputs.scale,
                    )
                    self.assertTrue(torch.equal(raw.cpu(), expected))
                    self.assertTrue(torch.equal(controlled.cpu(), expected.bfloat16()))
                    self.assertTrue(torch.equal(ordinary.cpu(), expected.bfloat16()))
                    splits, length = _split_k_config(
                        1,
                        32,
                        topk,
                        torch.cuda.get_device_properties(0).multi_processor_count,
                    )
                    print(
                        f"saturated weighted score0/-1024 H32 K={topk} splits={splits} len={length} changed={changed}: exact scoped PASS; general weighted exp unqualified"
                    )


if __name__ == "__main__":
    unittest.main()
