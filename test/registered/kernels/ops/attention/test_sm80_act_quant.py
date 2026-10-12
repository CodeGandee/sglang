"""SM80 DSA activation encoding and CUDA graph regression checks."""

from __future__ import annotations

import math
import unittest
from collections.abc import Callable
from unittest.mock import patch

import torch
from sglang.kernels.ops.attention.dsa import triton_kernel as kernel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


def _decode(code: int) -> float:
    magnitude = code & 127
    if magnitude == 127:
        return math.nan
    exponent, fraction = divmod(magnitude, 8)
    value = (
        math.ldexp(fraction, -9)
        if exponent == 0
        else math.ldexp(8 + fraction, exponent - 10)
    )
    return math.copysign(value, -1 if code & 128 else 1)


def _encode(value: float) -> int:
    # Independent finite nearest-representation encoding, ties to even.
    if not math.isfinite(value) or abs(value) > 448:
        raise ValueError("reference covers finite normalized inputs within +/-448")
    sign = 128 if math.copysign(1, value) < 0 else 0
    code = min(range(127), key=lambda code: (abs(_decode(code) - abs(value)), code & 1))
    return sign | code


def _samples() -> list[float]:
    values = [_decode(code) for code in range(256) if code & 127 != 127]
    values.extend((_decode(code) + _decode(code + 1)) / 2 for code in range(126))
    values.extend(-value for value in values[254:])
    return values


def _fixture(
    rows: int, heads: int, groups: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    samples = _samples()
    values: list[float] = []
    codes: list[int] = []
    scales: list[float] = []
    for group in range(rows * heads * groups):
        scale = math.ldexp(1.0, group % 7 - 3)
        normalized = [448.0] + [
            samples[(group * 127 + column) % len(samples)] for column in range(127)
        ]
        values.extend(value * scale for value in normalized)
        codes.extend(_encode(value) for value in normalized)
        scales.append(scale)
    shape = (rows, heads, groups * 128)
    return (
        torch.tensor(values, dtype=torch.bfloat16).reshape(shape),
        torch.tensor(codes, dtype=torch.uint8).reshape(shape),
        torch.tensor(scales, dtype=torch.float32).reshape(rows, heads, groups),
    )


class TestSM80ActivationEncodingCPU(CustomTestCase):
    def test_independent_finite_encoder_matches_torch_cpu(self) -> None:
        samples = _samples()
        expected = torch.tensor(
            [_encode(value) for value in samples], dtype=torch.uint8
        )
        actual = torch.tensor(samples, dtype=torch.float32).to(torch.float8_e4m3fn)
        self.assertTrue(torch.equal(actual.view(torch.uint8), expected))
        for code in range(256):
            if code & 127 != 127:
                self.assertEqual(_encode(_decode(code)), code)

    def test_sm80_only_temporary_selection(self) -> None:
        with patch.object(
            torch.cuda, "get_device_capability", return_value=(8, 0)
        ) as cap:
            self.assertTrue(kernel._act_quant_uses_fp32_output(torch.device("cuda:1")))
            cap.assert_called_once_with(torch.device("cuda:1"))
        for capability in ((8, 6), (8, 9), (9, 0), (10, 0)):
            with patch.object(
                torch.cuda, "get_device_capability", return_value=capability
            ):
                self.assertFalse(
                    kernel._act_quant_uses_fp32_output(torch.device("cuda"))
                )
        with patch.object(torch.cuda, "get_device_capability") as cap:
            self.assertFalse(kernel._act_quant_uses_fp32_output(torch.device("cpu")))
            cap.assert_not_called()

    def test_actual_entrypoint_preserves_other_arch_direct_store(self) -> None:
        writes: list[torch.dtype] = []

        class Launch:
            def __getitem__(self, grid: object) -> Callable[..., None]:
                def run(
                    x: torch.Tensor,
                    y: torch.Tensor,
                    scales: torch.Tensor,
                    *args: int,
                    **kwargs: object,
                ) -> None:
                    writes.append(y.dtype)
                    y.copy_(x)
                    scales.fill_(1)

                return run

        x = torch.ones(1, 128, dtype=torch.bfloat16)
        with (
            patch.object(kernel, "_act_quant_uses_fp32_output", return_value=False),
            patch.object(kernel, "_act_quant_kernel", Launch()),
        ):
            output, scales = kernel.act_quant(x)
        self.assertEqual(writes, [torch.float8_e4m3fn])
        self.assertEqual(output.dtype, torch.float8_e4m3fn)
        self.assertEqual(scales.dtype, torch.float32)
        self.assertTrue(
            torch.equal(
                output.view(torch.uint8), torch.full((1, 128), 56, dtype=torch.uint8)
            )
        )


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0),
    "requires SM80 CUDA",
)
class TestSM80ActivationEncodingGPU(CustomTestCase):
    def test_actual_entrypoint_exact_quantization_both_scale_modes(self) -> None:
        for rows, heads, groups in ((1, 1, 1), (33, 2, 2)):
            source, expected, scales_expected = _fixture(rows, heads, groups)
            x = source.cuda()
            for scale_fmt in (None, "ue8m0"):
                with self.subTest(
                    rows=rows, heads=heads, groups=groups, scale_fmt=scale_fmt
                ):
                    output, scales = kernel.act_quant(x, 128, scale_fmt)
                    self.assertEqual(output.shape, source.shape)
                    self.assertEqual(output.dtype, torch.float8_e4m3fn)
                    self.assertEqual(scales.dtype, torch.float32)
                    self.assertTrue(
                        torch.equal(output.view(torch.uint8).cpu(), expected)
                    )
                    self.assertTrue(torch.equal(scales.cpu(), scales_expected))

    def test_cuda_graph_replay_uses_changed_input(self) -> None:
        source, expected, scales_expected = _fixture(3, 2, 1)
        for scale_fmt in (None, "ue8m0"):
            x = source.cuda()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    kernel.act_quant(x, 128, scale_fmt)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output, scales = kernel.act_quant(x, 128, scale_fmt)
            graph.replay()
            self.assertTrue(torch.equal(output.view(torch.uint8).cpu(), expected))
            self.assertTrue(torch.equal(scales.cpu(), scales_expected))
            x.copy_(-source.cuda() / 2)
            eager_output, eager_scales = kernel.act_quant(x, 128, scale_fmt)
            graph.replay()
            self.assertTrue(
                torch.equal(output.view(torch.uint8), eager_output.view(torch.uint8))
            )
            self.assertTrue(torch.equal(scales, eager_scales))
            self.assertTrue(torch.equal(output.view(torch.uint8).cpu(), expected ^ 128))
            self.assertTrue(torch.equal(scales.cpu(), scales_expected / 2))


if __name__ == "__main__":
    unittest.main()
