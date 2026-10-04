"""Run actual Q-B constructor eligibility and dispatch without model allocation."""

import ast
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.linear import ColumnParallelLinear
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _actual_constructor_and_forward():
    """Compile the owning constructor's exact four flag statements and method.

    The full constructor allocates model projections and needs distributed/GPU
    state. Executing its real eligibility block against actual linear modules
    isolates this constructor failure without replacing or copying its predicate.
    """
    source = (
        Path(__file__).resolve().parents[4] / "python/sglang/srt/models/deepseek_v2.py"
    )
    module = ast.parse(source.read_text())
    attention = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "DeepseekV2AttentionMLA"
    )
    constructor = next(
        node
        for node in attention.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    flags = {
        "has_q_b_proj",
        "q_b_proj_verified_shapes",
        "_q_b_proj_verified_shape",
        "_use_min_latency_q_b_gemm",
    }
    selected = []
    for node in constructor.body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        if any(
            isinstance(target, ast.Name)
            and target.id in flags
            or isinstance(target, ast.Attribute)
            and target.attr in flags
            for target in targets
        ):
            selected.append(copy.deepcopy(node))
    if len(selected) != 4:
        raise AssertionError("Q-B constructor eligibility block changed")
    probe = ast.parse("def initialize(self):\n    pass\n").body[0]
    probe.body = selected
    forward = copy.deepcopy(
        next(
            node
            for node in attention.body
            if isinstance(node, ast.FunctionDef) and node.name == "q_b_proj_forward"
        )
    )
    compiled = ast.Module(
        body=[ast.parse("from __future__ import annotations").body[0], probe, forward],
        type_ignores=[],
    )
    namespace = {}
    exec(compile(ast.fix_missing_locations(compiled), str(source), "exec"), namespace)
    return namespace


def _linear(shape=None, *, packed=False):
    layer = ColumnParallelLinear.__new__(ColumnParallelLinear)
    torch.nn.Module.__init__(layer)
    name = "weight_packed" if packed else "weight"
    layer.register_parameter(
        name, torch.nn.Parameter(torch.empty(shape, device="meta"), requires_grad=False)
    )
    return layer


class TestQBProjectionEligibility(unittest.TestCase):
    def test_packed_projection_has_no_ordinary_weight_and_is_ineligible(self):
        layer = _linear((2048, 2048), packed=True)
        self.assertFalse(hasattr(layer, "weight"))
        attention = SimpleNamespace(q_b_proj=layer)
        _actual_constructor_and_forward()["initialize"](attention)
        self.assertTrue(attention.has_q_b_proj)
        self.assertFalse(attention._q_b_proj_verified_shape)

    def test_absent_projection_is_ineligible(self):
        attention = SimpleNamespace()
        _actual_constructor_and_forward()["initialize"](attention)
        self.assertFalse(attention.has_q_b_proj)
        self.assertFalse(attention._q_b_proj_verified_shape)

    def test_unverified_weight_shape_is_ineligible(self):
        attention = SimpleNamespace(q_b_proj=_linear((128, 128)))
        _actual_constructor_and_forward()["initialize"](attention)
        self.assertTrue(attention.has_q_b_proj)
        self.assertFalse(attention._q_b_proj_verified_shape)

    def test_both_verified_ordinary_weight_shapes_remain_eligible(self):
        for shape in ((2048, 2048), (4096, 2048)):
            with self.subTest(shape=shape):
                attention = SimpleNamespace(q_b_proj=_linear(shape))
                _actual_constructor_and_forward()["initialize"](attention)
                self.assertTrue(attention._q_b_proj_verified_shape)

    def test_packed_projection_uses_actual_default_dispatch_without_fused_probe(self):
        functions = _actual_constructor_and_forward()
        attention = SimpleNamespace(
            q_b_proj=_linear((2048, 2048), packed=True),
            num_local_heads=1,
            qk_head_dim=4,
        )
        functions["initialize"](attention)
        functions["fused_a_gemm_weight_eligible"] = Mock(
            side_effect=AssertionError(
                "packed weights must not be checked by fused eligibility"
            )
        )
        functions["linear_with_fused_a_gemm"] = Mock(
            side_effect=AssertionError(
                "packed weights must retain ordinary quantized dispatch"
            )
        )
        query = torch.arange(4, dtype=torch.float32).reshape(1, 4)
        with patch.object(
            ColumnParallelLinear, "forward", return_value=(query, None)
        ) as default:
            output = functions["q_b_proj_forward"](attention, query)
        default.assert_called_once_with(query)
        functions["fused_a_gemm_weight_eligible"].assert_not_called()
        self.assertFalse(attention._use_min_latency_q_b_gemm)
        self.assertTrue(torch.equal(output, query.view(1, 1, 4)))

    def test_verified_shape_still_requires_existing_fused_weight_eligibility(self):
        functions = _actual_constructor_and_forward()
        attention = SimpleNamespace(
            q_b_proj=_linear((2048, 2048)), num_local_heads=1, qk_head_dim=4
        )
        functions["initialize"](attention)
        functions["fused_a_gemm_weight_eligible"] = Mock(return_value=False)
        functions["linear_with_fused_a_gemm"] = Mock(
            side_effect=AssertionError(
                "ineligible ordinary weights must retain default dispatch"
            )
        )
        query = torch.arange(4, dtype=torch.float32).reshape(1, 4)
        with patch.object(
            ColumnParallelLinear, "forward", return_value=(query, None)
        ) as default:
            output = functions["q_b_proj_forward"](attention, query)
        functions["fused_a_gemm_weight_eligible"].assert_called_once_with(
            attention.q_b_proj
        )
        default.assert_called_once_with(query)
        self.assertTrue(torch.equal(output, query.view(1, 1, 4)))

    def test_verified_eligible_ordinary_weights_preserve_fused_dispatch(self):
        functions = _actual_constructor_and_forward()
        attention = SimpleNamespace(
            q_b_proj=_linear((4096, 2048)),
            num_local_heads=1,
            qk_head_dim=4,
            fused_a_gemm_backend="auto",
        )
        functions["initialize"](attention)
        query = torch.arange(4, dtype=torch.float32).reshape(1, 4)
        functions["fused_a_gemm_weight_eligible"] = Mock(return_value=True)
        functions["linear_with_fused_a_gemm"] = Mock(return_value=query)
        with patch.object(
            ColumnParallelLinear,
            "forward",
            side_effect=AssertionError(
                "eligible ordinary weights should retain fused path"
            ),
        ):
            output = functions["q_b_proj_forward"](attention, query)
        functions["linear_with_fused_a_gemm"].assert_called_once_with(
            attention.q_b_proj, query, backend="auto"
        )
        self.assertTrue(torch.equal(output, query.view(1, 1, 4)))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
