"""Scoped SM80 TileLang routing/padding regressions; weighted parity is diagnostic."""

from __future__ import annotations

import ast
import unittest
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch

import torch
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")


def _maintained_wrapper(v1: Mock, v2: Mock) -> Callable[..., torch.Tensor]:
    # Importing TileLang requires its compiler installation. CPU routing checks
    # execute the actual maintained wrapper with only its factories substituted.
    path = (
        Path(__file__).parents[5]
        / "python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py"
    )
    tree = ast.parse(path.read_text())
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "tilelang_sparse_fwd"
    )
    namespace: dict[str, object] = {
        "torch": torch,
        "_is_hip": False,
        "sparse_attention_fwd_kernel_v1": v1,
        "sparse_attention_fwd_kernel_v2": v2,
    }
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 - trusted tracked source
    return cast(Callable[..., torch.Tensor], namespace["tilelang_sparse_fwd"])


def _fake_tensor(shape: tuple[int, ...], dtype: torch.dtype) -> SimpleNamespace:
    tensor = SimpleNamespace(shape=shape, dtype=dtype, device=torch.device("cuda"))
    tensor.dim = Mock(return_value=3)
    tensor.contiguous = Mock(return_value=tensor)
    tensor.unsqueeze = Mock(return_value=tensor)
    return tensor


def _native_forward(
    q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

    # Exercise the real DSA method and its selected-row unsqueeze, not just v1.
    return DeepseekSparseAttnBackend._forward_tilelang(
        cast(DeepseekSparseAttnBackend, None), q, kv, 512, indices[:, 0], 0.0625
    )


def _logical_reference(
    q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    query, pool, selected = q.cpu().double(), kv.cpu().double(), indices.cpu()
    result = torch.zeros((1, *query.shape[:2], 512), dtype=torch.float64)
    for row in range(query.shape[0]):
        physical = selected[row, 0]
        physical = physical[(physical >= 0) & (physical < pool.shape[0])].long()
        if physical.numel():
            keys = pool[physical, 0]
            logits = query[row] @ keys.T * 0.0625
            result[0, row] = logits.softmax(dim=-1) @ keys[:, :512]
    return result


class TestSM80TileLangDispatch(CustomTestCase):
    def test_sm80_selects_single_stage_v1_and_compacts_views(self) -> None:
        for heads in (8, 32):
            with self.subTest(heads=heads):
                result = torch.zeros(1)
                launch = Mock(return_value=result)
                v1, v2 = Mock(return_value=launch), Mock()
                tensors = (
                    _fake_tensor((3, heads, 576), torch.bfloat16),
                    _fake_tensor((4, 1, 576), torch.bfloat16),
                    _fake_tensor((3, 1, 2048), torch.int32),
                )
                with patch("torch.cuda.get_device_capability", return_value=(8, 0)):
                    self.assertIs(_maintained_wrapper(v1, v2)(*tensors, 0.0625), result)
                v1.assert_called_once_with(
                    heads, 512, 64, 2048, sm_scale=0.0625, num_stages=1
                )
                v2.assert_not_called()
                launch.assert_called_once_with(*tensors)
                for tensor in tensors:
                    tensor.contiguous.assert_called_once_with()

    def test_sm80_effective_h64_tile_uses_smaller_existing_index_block(self) -> None:
        for heads in (64, 128):
            with self.subTest(heads=heads):
                result = torch.zeros(1)
                launch = Mock(return_value=result)
                v1, v2 = Mock(return_value=launch), Mock()
                tensors = (
                    _fake_tensor((3, heads, 576), torch.bfloat16),
                    _fake_tensor((4, 1, 576), torch.bfloat16),
                    _fake_tensor((3, 1, 2048), torch.int32),
                )
                with patch("torch.cuda.get_device_capability", return_value=(8, 0)):
                    self.assertIs(_maintained_wrapper(v1, v2)(*tensors, 0.0625), result)
                v1.assert_called_once_with(
                    heads, 512, 64, 2048, sm_scale=0.0625, block_I=16, num_stages=1
                )
                v2.assert_not_called()
                launch.assert_called_once_with(*tensors)
                for tensor in tensors:
                    tensor.contiguous.assert_called_once_with()

    def test_other_cuda_architectures_preserve_v2(self) -> None:
        for capability in ((9, 0), (10, 0)):
            with self.subTest(capability=capability):
                launch = Mock(return_value=torch.zeros(1))
                v1, v2 = Mock(), Mock(return_value=launch)
                tensors = (
                    _fake_tensor((2, 64, 576), torch.bfloat16),
                    _fake_tensor((4, 1, 576), torch.bfloat16),
                    _fake_tensor((2, 1, 2048), torch.int32),
                )
                with patch("torch.cuda.get_device_capability", return_value=capability):
                    _maintained_wrapper(v1, v2)(*tensors, 0.0625)
                v2.assert_called_once_with(64, 512, 64, 2048, sm_scale=0.0625)
                v1.assert_not_called()
                for tensor in tensors:
                    tensor.contiguous.assert_not_called()

    def test_sm80_rejects_non_bf16_before_launch(self) -> None:
        for query_dtype, cache_dtype in (
            (torch.float32, torch.bfloat16),
            (torch.bfloat16, torch.float8_e4m3fn),
        ):
            v1, v2 = Mock(), Mock()
            tensors = (
                _fake_tensor((2, 8, 576), query_dtype),
                _fake_tensor((4, 1, 576), cache_dtype),
                _fake_tensor((2, 1, 2048), torch.int32),
            )
            with (
                patch("torch.cuda.get_device_capability", return_value=(8, 0)),
                self.assertRaisesRegex(ValueError, "requires BF16"),
            ):
                _maintained_wrapper(v1, v2)(*tensors, 0.0625)
            v1.assert_not_called()
            v2.assert_not_called()


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0),
    "requires an actual SM80 GPU and the native TileLang installation",
)
class TestSM80TileLangPrefill(CustomTestCase):
    def test_native_empty_prefix_padding_and_head_partitions(self) -> None:
        for heads in (8, 32, 64):
            with self.subTest(heads=heads):
                # Admitted sliced layouts are compacted only by the SM80 branch.
                q = torch.zeros(
                    (3, heads * 2, 576), device="cuda", dtype=torch.bfloat16
                )[:, ::2]
                kv = torch.zeros((4, 1, 1152), device="cuda", dtype=torch.bfloat16)[
                    ..., :576
                ]
                kv.copy_(torch.arange(1, 5, device="cuda")[:, None, None])
                indices = torch.full(
                    (3, 1, 4096), -1, device="cuda", dtype=torch.int32
                )[..., :2048]
                # A completely invalid first tile precedes an exactly uniform
                # valid tile. Out-of-range positive rows must add no denominator.
                indices[0, 0, 64:128] = torch.arange(64, device="cuda") % 2
                indices[0, 0, 128:192] = kv.shape[0]
                indices[1, 0, 0] = 2
                indices[2, 0, 64] = -7
                expected = torch.empty((1, 3, heads, 512), dtype=torch.bfloat16)
                expected[:, 0].fill_(1.5)
                expected[:, 1].fill_(3)
                expected[:, 2].zero_()
                output = _native_forward(q, kv, indices)
                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertTrue(torch.isfinite(output).all().item())
                self.assertTrue(torch.equal(output.cpu(), expected))
                # Reuse the same shapes with changed selected physical rows.
                indices[1, 0, 0] = 0
                expected[:, 1].fill_(1)
                self.assertTrue(
                    torch.equal(_native_forward(q, kv, indices).cpu(), expected)
                )

    def test_native_h64_full_topk_and_changed_selection(self) -> None:
        q = torch.zeros((2, 64, 576), device="cuda", dtype=torch.bfloat16)
        kv = torch.empty((2048, 1, 576), device="cuda", dtype=torch.bfloat16)
        rows = torch.arange(2048, device="cuda", dtype=torch.int32)
        kv.copy_((rows.remainder(2) + 1)[:, None, None])
        indices = torch.full((2, 1, 2048), -1, device="cuda", dtype=torch.int32)
        indices[0, 0] = rows
        properties = torch.cuda.get_device_properties(q.device)
        print(
            "SM80 native H64 prefill: "
            f"device={q.device}, shared_memory_per_block_optin="
            f"{properties.shared_memory_per_block_optin}, topk=2048"
        )
        expected = torch.zeros((1, 2, 64, 512), dtype=torch.bfloat16)
        expected[:, 0].fill_(1.5)
        output = _native_forward(q, kv, indices)
        self.assertTrue(torch.isfinite(output).all().item())
        self.assertTrue(torch.equal(output.cpu(), expected))

        # The same admitted shapes now gather only even rows (all ones), while
        # the previously empty row gathers exactly one odd row (all twos).
        indices[0, 0] = rows.remainder(1024) * 2
        indices[1, 0, 0] = 2047
        expected[:, 0].fill_(1)
        expected[:, 1].fill_(2)
        self.assertTrue(torch.equal(_native_forward(q, kv, indices).cpu(), expected))

    def test_weighted_logical_reference_diagnostic(self) -> None:
        generator = torch.Generator().manual_seed(91)
        q = (
            (torch.randn((2, 8, 576), generator=generator) / 8)
            .to(torch.bfloat16)
            .cuda()
        )
        kv = (
            (torch.randn((67, 1, 576), generator=generator) / 8)
            .to(torch.bfloat16)
            .cuda()
        )
        indices = torch.full((2, 1, 2048), -1, dtype=torch.int32, device="cuda")
        indices[0, 0, 64:128] = torch.arange(64, device="cuda")
        indices[1, 0, 0] = 66
        reference = _logical_reference(q, kv, indices)
        output = _native_forward(q, kv, indices).cpu()
        self.assertTrue(torch.isfinite(output).all().item())
        # One selected key is an exact arithmetic witness for arbitrary q/kv.
        self.assertTrue(torch.equal(output[0, 1], kv[66, 0, :512].cpu().expand(8, 512)))
        mismatch = int((output != reference.to(torch.bfloat16)).sum())
        residual = float((output.double() - reference).abs().max())
        print(
            f"weighted diagnostic: max_abs={residual:.9g}, strict_bf16_mismatches={mismatch}; "
            "FP64 logical reference does not qualify existing BF16 probabilities/exp2 semantics"
        )


if __name__ == "__main__":
    unittest.main()
