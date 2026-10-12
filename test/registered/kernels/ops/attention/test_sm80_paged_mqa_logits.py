"""Bounded SM80 indexer qualification, without model weights or profilers.

The scalar reference decodes bytes and addresses logical tokens independently
of the tensor reference and candidate. Constructed dyadic fixtures admit exact
FP64 arithmetic: their bounded integer sums fit within the 53-bit significand.
Only the candidate's explicit FP64 specialization qualifies its shared semantic
boundary. Production FP32 accumulation and graph reproducibility are separate.
"""

from __future__ import annotations

import math
import struct
import unittest
from dataclasses import dataclass
from unittest.mock import patch

import torch
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b", runner_config="1-gpu-small")

_PAGE = 64
_DIM = 128
_KEY_BYTES = _PAGE * _DIM
_PAGE_BYTES = _PAGE * (_DIM + 4)


@dataclass(frozen=True)
class _Inputs:
    q: torch.Tensor
    cache: torch.Tensor
    weights: torch.Tensor
    lengths: torch.Tensor
    table: torch.Tensor
    width: int

    def on(self, device: str) -> _Inputs:
        return _Inputs(
            self.q.to(device),
            self.cache.to(device),
            self.weights.to(device),
            self.lengths.to(device),
            self.table.to(device),
            self.width,
        )


def _decode_byte(bits: int) -> float:
    # Independent scalar implementation: no candidate decoder or torch FP8 cast.
    magnitude = bits & 127
    if magnitude == 127:
        return math.nan
    exponent, fraction = divmod(magnitude, 8)
    unsigned = (
        math.ldexp(float(fraction), -9)
        if exponent == 0
        else math.ldexp(float(8 + fraction), exponent - 10)
    )
    return math.copysign(unsigned, -1.0 if bits & 128 else 1.0)


def _pack(keys: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    raw = torch.empty(keys.shape[0], _PAGE_BYTES, dtype=torch.uint8)
    raw[:, :_KEY_BYTES] = keys.contiguous().view(torch.uint8).flatten(1)
    raw[:, _KEY_BYTES:] = scales.contiguous().view(torch.uint8).flatten(1)
    return raw.view(-1, _PAGE, 1, _DIM + 4)


def _fixture(lengths: list[int], heads: int, *, dense: bool = False) -> _Inputs:
    # Exact powers of two, integer sums and modest exponents make CPU/GPU FP64
    # references exact, without relying on repeated approximate agreement.
    width = max(lengths)
    pages_per_request = (width + _PAGE - 1) // _PAGE
    pages = len(lengths) * pages_per_request
    generator = torch.Generator().manual_seed(42)
    if dense:
        keys = torch.randint(-8, 9, (pages, _PAGE, _DIM), generator=generator).float()
        keys *= 1 / 8
        q = torch.randint(-8, 9, (len(lengths), heads, _DIM), generator=generator)
        q = (q.float() / 8).to(torch.float8_e4m3fn)
        weights = torch.randint(-4, 5, (len(lengths), heads), generator=generator)
        weights = weights.float() / 8
        scales = torch.randint(1, 5, (pages, _PAGE), generator=generator).float() / 4
    else:
        # Single nonzero dimension isolates page/scale addressing. Monotone
        # per-token scales provide a unique top-k boundary even at length 2049.
        keys = torch.zeros(pages, _PAGE, _DIM)
        keys[..., 0] = 1
        q = torch.zeros(len(lengths), heads, _DIM).to(torch.float8_e4m3fn)
        q.view(torch.uint8)[..., 0] = 0x38  # +1
        weights = torch.ones(len(lengths), heads, dtype=torch.float32)
        scales = torch.arange(1, pages * _PAGE + 1).view(pages, _PAGE).float()
    table = torch.arange(pages, dtype=torch.int32).view(len(lengths), -1).flip(1)
    return _Inputs(
        q,
        _pack(keys.to(torch.float8_e4m3fn), scales),
        weights,
        torch.tensor(lengths, dtype=torch.int32),
        table.contiguous(),
        width,
    )


def _scalar_reference(inputs: _Inputs) -> torch.Tensor:
    # Byte offset and little-endian scale reconstruction differ from the tensor
    # oracle's reshape/view/gather arithmetic. Traverse each logical token.
    raw_pages = inputs.cache.view(torch.uint8).flatten(1).tolist()
    raw_q = inputs.q.view(torch.uint8).tolist()
    weights = inputs.weights.tolist()
    table = inputs.table.tolist()
    result = torch.zeros(len(raw_q), inputs.width, dtype=torch.float64)
    for row, length in enumerate(inputs.lengths.tolist()):
        for logical in range(min(length, inputs.width)):
            page_index, offset = divmod(logical, _PAGE)
            if page_index >= len(table[row]):
                continue
            physical = table[row][page_index]
            if not 0 <= physical < len(raw_pages):
                continue
            raw = raw_pages[physical]
            scale_begin = _KEY_BYTES + offset * 4
            scale = struct.unpack("<f", bytes(raw[scale_begin : scale_begin + 4]))[0]
            terms = []
            for head, query in enumerate(raw_q[row]):
                dot = math.fsum(
                    _decode_byte(query[d]) * _decode_byte(raw[offset * _DIM + d])
                    for d in range(_DIM)
                )
                terms.append(
                    (math.nan if math.isnan(dot) else max(dot, 0.0))
                    * weights[row][head]
                )
            result[row, logical] = math.fsum(terms) * scale
    return result


def _tensor_reference(inputs: _Inputs) -> torch.Tensor:
    # Native CPU float8 conversion supplies an independent decode implementation.
    raw = inputs.cache.view(torch.uint8).flatten(1)
    keys = raw[:, :_KEY_BYTES].contiguous().view(torch.float8_e4m3fn)
    keys = keys.reshape(-1, _PAGE, _DIM).double()
    scales = raw[:, _KEY_BYTES:].contiguous().view(torch.float32).double()
    q, weights = inputs.q.double(), inputs.weights.double()
    out = torch.zeros(q.shape[0], inputs.width, dtype=torch.float64)
    for row in range(q.shape[0]):
        positions = torch.arange(min(int(inputs.lengths[row]), inputs.width))
        if not positions.numel():
            continue
        page_numbers = positions // _PAGE
        in_table = page_numbers < inputs.table.shape[1]
        physical = inputs.table[
            row, page_numbers.clamp(max=inputs.table.shape[1] - 1)
        ].long()
        valid = in_table & (physical >= 0) & (physical < keys.shape[0])
        positions, physical = positions[valid], physical[valid]
        offsets = positions.remainder(_PAGE)
        dots = (q[row] @ keys[physical, offsets].T).relu()
        out[row, positions] = (dots * weights[row, :, None]).sum(0) * scales[
            physical, offsets
        ]
    return out


def _kernel(inputs: _Inputs, *, fp64: bool = False) -> torch.Tensor:
    from sglang.kernels.ops.attention.dsa.triton_paged_mqa_logits_sm80 import (
        triton_paged_mqa_logits,
    )

    return triton_paged_mqa_logits(
        inputs.q,
        inputs.cache,
        inputs.weights,
        inputs.lengths,
        inputs.table,
        inputs.width,
        _verification_fp64=fp64,
    )


class TestSM80ReferenceQualification(CustomTestCase):
    """Qualify small independent references without launching a CUDA operation."""

    def test_all_byte_codes_and_signed_zero(self) -> None:
        raw = torch.arange(256, dtype=torch.uint8)
        tensor_values = raw.view(torch.float8_e4m3fn).float()
        for code, actual in enumerate(tensor_values.tolist()):
            expected = _decode_byte(code)
            if math.isnan(expected):
                self.assertTrue(math.isnan(actual))
            else:
                self.assertEqual(struct.pack("<f", expected), struct.pack("<f", actual))

    def test_independent_address_and_arithmetic_references(self) -> None:
        for dense in (False, True):
            inputs = _fixture([63, 65, 1, 64], 4, dense=dense)
            self.assertTrue(
                torch.equal(_scalar_reference(inputs), _tensor_reference(inputs))
            )
        # Explicit physical address sentinel: request 0's first logical token
        # reads physical page 1, scale 65; request 1 reads physical page 3.
        sentinel = _fixture([65, 65], 1)
        reference = _scalar_reference(sentinel)
        self.assertEqual(float(reference[0, 0]), 65)
        self.assertEqual(float(reference[1, 0]), 193)
        self.assertEqual(float(reference[0, 64]), 1)

    def test_excluded_modes_reject_before_launch(self) -> None:
        from sglang.kernels.ops.attention.dsa.paged_mqa_logits import (
            sm80_paged_mqa_logits,
        )

        cpu = _fixture([65], 1)
        for mode in ("prefill", "chunked-prefill", "target-verify"):
            with self.subTest(mode=mode), self.assertRaises(NotImplementedError):
                sm80_paged_mqa_logits(
                    cpu.q,
                    cpu.cache,
                    cpu.weights,
                    cpu.lengths,
                    cpu.table,
                    cpu.width,
                    q_offset=1,
                    B=1,
                    next_n=1,
                    is_decode=False,
                )
        for extra in ({"next_n": 2}, {"indices": torch.tensor([0])}):
            keywords = {"q_offset": 1, "B": 1, "next_n": 1, "is_decode": True}
            keywords.update(extra)
            with self.assertRaises(NotImplementedError):
                sm80_paged_mqa_logits(
                    cpu.q,
                    cpu.cache,
                    cpu.weights,
                    cpu.lengths,
                    cpu.table,
                    cpu.width,
                    **keywords,
                )

    def test_explicit_sm80_selection_and_other_device_controls(self) -> None:
        from sglang.srt.layers.attention.dsa import paged_mqa_logits_backend as backend

        for capability in ((8, 0), (9, 0), (10, 0)):
            with (
                patch.object(backend, "is_hip", return_value=False),
                patch.object(backend, "is_cuda", return_value=True),
                patch.object(backend, "get_device_capability", return_value=capability),
            ):
                self.assertEqual(
                    backend.DSAPagedMQALogitsBackend.resolve("auto"),
                    backend.DSAPagedMQALogitsBackend.DEEPGEMM,
                )
                if capability == (8, 0):
                    self.assertTrue(
                        backend.DSAPagedMQALogitsBackend.resolve(
                            "triton-sm80"
                        ).is_triton_sm80()
                    )
                else:
                    with self.assertRaisesRegex(ValueError, "requires SM80"):
                        backend.DSAPagedMQALogitsBackend.resolve("triton-sm80")
        with (
            patch.object(backend, "is_hip", return_value=True),
            self.assertRaises(ValueError),
        ):
            backend.DSAPagedMQALogitsBackend.resolve("triton-sm80")


class TestSM80PagedMQALogits(CustomTestCase):
    """Exercise actual SM80 production and controlled-arithmetic kernels."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            8,
            0,
        ):
            raise unittest.SkipTest(
                "SM80 GPU required; no other architecture qualifies this kernel"
            )

    def test_actual_byte_decoder_all_encodings(self) -> None:
        from sglang.kernels.ops.attention.dsa.triton_paged_mqa_logits_sm80 import (
            triton_decode_e4m3fn,
        )

        actual = triton_decode_e4m3fn(
            torch.arange(256, dtype=torch.uint8, device="cuda")
        ).cpu()
        for code, value in enumerate(actual.tolist()):
            expected = _decode_byte(code)
            if math.isnan(expected):
                self.assertTrue(math.isnan(value))
            else:
                self.assertEqual(struct.pack("<f", value), struct.pack("<f", expected))

    def test_pages_lengths_head_tiles_and_isolation(self) -> None:
        for length in (1, 63, 64, 65, 2047, 2048, 2049, 4095, 4096, 4097, 16384):
            for heads in (1, 8, 16, 32) if length == 65 else (4,):
                with self.subTest(length=length, heads=heads):
                    cpu = _fixture(
                        [length, max(0, length - 7), 0, max(1, length - 63)], heads
                    )
                    expected = _tensor_reference(cpu).float()
                    # Sparse one-dimension fixtures have exact bounded integer
                    # FP32 arithmetic, so strict equality checks real codegen.
                    actual = _kernel(cpu.on("cuda")).cpu()
                    self.assertTrue(torch.equal(actual, expected))

    def test_fp64_actual_candidate_shared_semantics(self) -> None:
        for heads in (1, 4, 17):
            for dense in (False, True):
                with self.subTest(heads=heads, dense=dense):
                    cpu = _fixture([1, 63, 64, 65], heads, dense=dense)
                    r1, r2 = _scalar_reference(cpu), _tensor_reference(cpu)
                    self.assertTrue(torch.equal(r1, r2))
                    # Exact dyadic reference sums fit within binary64 and the
                    # declared output boundary is FP32, not BF16.
                    controlled = _kernel(cpu.on("cuda"), fp64=True).cpu()
                    self.assertTrue(torch.equal(controlled, r1.float()))
                    production = _kernel(cpu.on("cuda")).cpu()
                    self.assertTrue(torch.equal(production, r1.float()))
                    print(
                        f"controlled-fp64 H={heads} dense={dense}: exact independent references and FP32 casts PASS"
                    )

    def test_invalid_page_masks_and_unavailable_table_entries(self) -> None:
        cpu = _fixture([129, 65], 1)
        cpu.table[0, 0] = -1
        cpu.table[0, 1] = cpu.cache.shape[0] + 10
        short = _Inputs(
            cpu.q, cpu.cache, cpu.weights, cpu.lengths, cpu.table[:, :2], cpu.width
        )
        expected = _scalar_reference(short).float()
        self.assertTrue(torch.equal(expected, _tensor_reference(short).float()))
        self.assertTrue(torch.equal(_kernel(short.on("cuda")).cpu(), expected))

    def test_actual_unfused_consumer_unique_cutoff_and_ties(self) -> None:
        from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend

        cpu = _fixture([2049, 63], 1)
        device = cpu.on("cuda")
        output = _kernel(device)
        controlled = _kernel(device, fp64=True)
        self.assertTrue(torch.equal(output, controlled))
        self.assertTrue(torch.equal(_scalar_reference(cpu), _tensor_reference(cpu)))
        selected = DSATopKBackend.TORCH.topk_func(output, device.lengths, 2048).cpu()
        controlled_selected = DSATopKBackend.TORCH.topk_func(
            controlled, device.lengths, 2048
        ).cpu()
        self.assertTrue(torch.equal(selected, controlled_selected))
        expected_scores = _tensor_reference(cpu)
        for row, length in enumerate(cpu.lengths.tolist()):
            wanted = sorted(
                range(length),
                key=lambda i: float(expected_scores[row, i]),
                reverse=True,
            )[:2048]
            actual = selected[row][selected[row] >= 0].tolist()
            self.assertEqual(set(actual), set(wanted))
            self.assertEqual(len(actual), len(set(actual)))
        # Tied scores do not imply a unique reference membership. Require valid
        # cardinality and cutoff scores; retain ordinary repeat parity separately.
        tied_scores = torch.ones(1, 65, device="cuda")
        tied_lengths = torch.tensor([63], dtype=torch.int32, device="cuda")
        first = DSATopKBackend.TORCH.topk_func(tied_scores, tied_lengths, 8)
        second = DSATopKBackend.TORCH.topk_func(tied_scores, tied_lengths, 8)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(len(set(first.cpu().flatten().tolist())), 8)
        self.assertTrue(bool(((first >= 0) & (first < 63)).all()))
        print(
            "ties: cardinality/membership valid; repeated native Torch selection exact; unique membership not claimed"
        )

    def test_nan_policy_and_empty_output(self) -> None:
        device = _fixture([1], 1).on("cuda")
        # Place a NaN in the represented key selected by physical page 0.
        device.cache.view(torch.uint8).flatten(1)[0, 0] = 0x7F
        self.assertTrue(bool(torch.isnan(_kernel(device)).all()))
        self.assertTrue(bool(torch.isnan(_kernel(device, fp64=True)).all()))
        # Zero-width and zero-request contracts require no launch, with the same
        # FP32 result dtype and exact empty shape.
        width_zero = _Inputs(
            device.q, device.cache, device.weights, device.lengths, device.table, 0
        )
        self.assertEqual(tuple(_kernel(width_zero).shape), (1, 0))
        rows_zero = _Inputs(
            device.q[:0],
            device.cache,
            device.weights[:0],
            device.lengths[:0],
            device.table[:0],
            65,
        )
        empty = _kernel(rows_zero)
        self.assertEqual(tuple(empty.shape), (0, 65))
        self.assertEqual(empty.dtype, torch.float32)

    def test_noncontiguous_admitted_views(self) -> None:
        cpu = _fixture([65, 63], 4)
        device = cpu.on("cuda")
        q_storage = torch.zeros(2, 8, _DIM, device="cuda", dtype=torch.float8_e4m3fn)
        q_storage[:, ::2] = device.q
        w_storage = torch.zeros(2, 8, device="cuda")
        w_storage[:, ::2] = device.weights
        table_storage = torch.zeros(2, 4, device="cuda", dtype=torch.int32)
        table_storage[:, ::2] = device.table
        strided = _Inputs(
            q_storage[:, ::2],
            device.cache,
            w_storage[:, ::2],
            device.lengths,
            table_storage[:, ::2],
            device.width,
        )
        self.assertTrue(
            torch.equal(_kernel(strided).cpu(), _tensor_reference(cpu).float())
        )

    def test_graph_replay_changes_pages_values_and_lengths(self) -> None:
        device = _fixture([65, 65], 4, dense=True).on("cuda")
        for _ in range(3):
            _kernel(device)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = _kernel(device)
        for iteration in range(3):
            device.table.copy_(device.table.flip(1))
            device.lengths.copy_(
                torch.tensor(
                    [63 + iteration, 65 - iteration], dtype=torch.int32, device="cuda"
                )
            )
            device.weights.mul_(-1)
            # Change query values without changing captured storage addresses.
            device.q.view(torch.uint8).bitwise_xor_(128)
            graph.replay()
            torch.cuda.synchronize()
            eager = _kernel(device)
            self.assertTrue(torch.equal(captured, eager))
            cpu = device.on("cpu")
            self.assertTrue(torch.equal(eager.cpu(), _tensor_reference(cpu).float()))

    def test_invalid_shapes_dtypes_and_layout_rejected(self) -> None:
        device = _fixture([65], 4).on("cuda")
        invalid = (
            _Inputs(
                device.q.float(),
                device.cache,
                device.weights,
                device.lengths,
                device.table,
                device.width,
            ),
            _Inputs(
                device.q[:, :, :64],
                device.cache,
                device.weights,
                device.lengths,
                device.table,
                device.width,
            ),
            _Inputs(
                device.q,
                device.cache,
                device.weights.half(),
                device.lengths,
                device.table,
                device.width,
            ),
            _Inputs(
                device.q,
                device.cache,
                device.weights,
                device.lengths.long(),
                device.table,
                device.width,
            ),
            _Inputs(
                device.q,
                device.cache,
                device.weights,
                device.lengths,
                device.table.long(),
                device.width,
            ),
            _Inputs(
                device.q, device.cache, device.weights, device.lengths, device.table, -1
            ),
            _Inputs(
                device.q,
                device.cache[..., :128],
                device.weights,
                device.lengths,
                device.table,
                device.width,
            ),
        )
        for case in invalid:
            with self.assertRaises((ValueError, TypeError)):
                _kernel(case)


if __name__ == "__main__":
    unittest.main()
