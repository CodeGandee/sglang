"""Provider-independent ShadowKV operation contract tests."""

import subprocess
import sys

import pytest
import torch
from sglang.kernels import registry
from sglang.kernels.ops import shadowkv
from sglang.kernels.ops.shadowkv import providers
from sglang.kernels.ops.shadowkv.providers import reference
from sglang.kernels.selector import (
    KernelSelectionError,
    KernelSelectionPolicy,
    select_kernel_candidate,
)
from sglang.kernels.spec import (
    CapabilityRequirement as Cap,
)
from sglang.kernels.spec import (
    KernelBackend,
    KernelExecutionProperties,
    KernelInputEnvelope,
    KernelSpec,
    KernelSpecialization,
    PlatformInfo,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _reconstruction_fixture(*, head_dim=128, rank=64):
    torch.manual_seed(20260831 + head_dim + rank)
    u = (torch.randn(17, rank) * 0.125).to(torch.bfloat16)
    sv = (torch.randn(2, rank, head_dim) * 0.125).to(torch.bfloat16)
    positions = torch.tensor([[0, 8, 16], [16, 8, 0]], dtype=torch.int64)
    return u, sv, positions


def _planner_fixture():
    return (
        torch.tensor([[1, 4, 8, -1], [2, 3, -1, -1]], dtype=torch.int64),
        torch.tensor([3, 2], dtype=torch.int32),
        torch.tensor([[8, 5, 4, 5], [2, 7, 3, 9]], dtype=torch.int64),
        torch.tensor([4, 4], dtype=torch.int32),
        torch.tensor([[5, 5], [3, 11]], dtype=torch.int64),
        torch.tensor([2, 2], dtype=torch.int32),
        torch.tensor([9, 8], dtype=torch.int64),
        torch.tensor([9, 10], dtype=torch.int64),
    )


def test_reconstruct_contract_preserves_inputs_and_caller_output():
    u, sv, positions = _reconstruction_fixture()
    frozen = tuple(tensor.clone() for tensor in (u, sv, positions))
    storage = torch.full((2, 5, 128), 91, dtype=torch.bfloat16)
    out = storage[:, :3]
    actual = shadowkv.reconstruct(u, sv, positions, out=out)
    expected = torch.einsum("hnr,hrd->hnd", u[positions], sv)
    assert actual is out
    assert torch.equal(actual, expected)
    assert torch.equal(storage[:, 3:], torch.full_like(storage[:, 3:], 91))
    for observed, original in zip((u, sv, positions), frozen, strict=True):
        assert torch.equal(observed, original)


def test_readable_reconstruct_supports_positive_unoptimized_head_dimension():
    u, sv, positions = _reconstruction_fixture(head_dim=96, rank=160)
    actual = shadowkv.reconstruct(u, sv, positions)
    expected = torch.einsum("hnr,hrd->hnd", u[positions], sv)
    assert actual.shape == (2, 3, 96)
    assert torch.equal(actual, expected)
    reference_spec = registry.get_implementation(
        shadowkv.RECONSTRUCT,
        shadowkv.REFERENCE_IMPLEMENTATIONS[shadowkv.RECONSTRUCT],
    )
    generic_spec = registry.get_implementation(
        shadowkv.RECONSTRUCT,
        shadowkv.GENERIC_AOT_IMPLEMENTATIONS[shadowkv.RECONSTRUCT],
    )
    request = KernelInputEnvelope(
        dtypes=frozenset({"bfloat16"}),
        head_dimensions=frozenset({96}),
        factor_ranks=frozenset({160}),
        features=frozenset({"current-stream", "caller-output-buffer"}),
    )
    assert reference_spec.input_envelope.supports(request)
    assert not generic_spec.input_envelope.supports(request)
    decision = select_kernel_candidate(
        shadowkv.RECONSTRUCT,
        policy=KernelSelectionPolicy.REFERENCE,
        platform=PlatformInfo(
            device_type="cuda",
            cuda_arch_major=8,
            cuda_arch_minor=0,
        ),
        envelope=request,
        qualified_implementation_ids=frozenset(
            {
                reference_spec.identity,
                generic_spec.identity,
            }
        ),
    )
    assert decision.selected.identity == reference_spec.identity


def test_reconstruct_rope_contract_matches_fp32_numerical_rule():
    u, sv, positions = _reconstruction_fixture(head_dim=64, rank=160)
    inverse = torch.linspace(1.0, 0.001, 32, dtype=torch.float32)
    actual = shadowkv.reconstruct_rope(u, sv, positions, inverse)
    reconstructed = torch.einsum("hnr,hrd->hnd", u[positions], sv).float()
    angles = positions.float().unsqueeze(-1) * inverse
    cosine = torch.cat((angles.cos(), angles.cos()), dim=-1)
    sine = torch.cat((angles.sin(), angles.sin()), dim=-1)
    rotated = torch.cat((-reconstructed[..., 32:], reconstructed[..., :32]), -1)
    expected = (reconstructed * cosine + rotated * sine).to(torch.bfloat16)
    assert torch.equal(actual, expected)


def test_packed_gqa_contract_mutates_only_declared_buffers():
    torch.manual_seed(20260831)
    query = (torch.randn(2, 8, 64) * 0.125).to(torch.bfloat16)
    keys = (torch.randn(2, 2, 17, 64) * 0.125).to(torch.bfloat16)
    values = (torch.randn(2, 2, 17, 64) * 0.125).to(torch.bfloat16)
    lengths = torch.tensor([0, 13], dtype=torch.int32)
    frozen = tuple(tensor.clone() for tensor in (query, keys, values, lengths))
    weights = torch.full((2, 8, 17), float("nan"), dtype=torch.float32)
    out = torch.full_like(query, 91)
    actual = shadowkv.packed_gqa(query, keys, values, lengths, weights=weights, out=out)
    assert actual is out
    assert torch.equal(out[0], torch.zeros_like(out[0]))
    assert torch.equal(weights[0], torch.zeros_like(weights[0]))
    assert torch.count_nonzero(weights[1, :, 13:]) == 0
    assert torch.isfinite(weights).all()
    for observed, original in zip((query, keys, values, lengths), frozen, strict=True):
        assert torch.equal(observed, original)


def test_plan_reuse_contract_is_discrete_deterministic_and_stable():
    arguments = _planner_fixture()
    first = shadowkv.plan_reuse(*arguments, max_reuse_chunks=3, chunk_size=8)
    second = shadowkv.plan_reuse(*arguments, max_reuse_chunks=3, chunk_size=8)
    assert torch.equal(first.plan, second.plan)
    assert torch.equal(
        first.deduplicated_exact_chunks, second.deduplicated_exact_chunks
    )
    assert torch.equal(first.counts, second.counts)
    assert first.kinds.tolist() == [[1, 0, 1, 0], [2, 2, 0, 2]]
    assert first.transfer_offsets.tolist() == [
        [-1, -1, -1, -1],
        [0, 8, -1, 16],
    ]
    assert first.counts.tolist() == [[2, 0, 1], [0, 3, 2]]


@pytest.mark.parametrize(
    "operation,mutation,message",
    [
        ("reconstruct", "dtype", "u must use"),
        ("reconstruct", "rank", "rank must be one of"),
        ("reconstruct_rope", "position", "positions exceed"),
        ("packed_gqa", "heads", "divisible"),
        ("plan_reuse", "chunk-size", "positive"),
    ],
)
def test_invalid_inputs_fail_before_provider_call(operation, mutation, message):
    called = False

    def provider(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("invalid input reached provider")

    with pytest.raises((TypeError, ValueError), match=message):
        if operation == "reconstruct":
            u, sv, positions = _reconstruction_fixture()
            if mutation == "dtype":
                u = u.float()
            else:
                u = torch.zeros((17, 96), dtype=torch.bfloat16)
                sv = torch.zeros((2, 96, 128), dtype=torch.bfloat16)
            shadowkv.reconstruct(u, sv, positions, implementation=provider)
        elif operation == "reconstruct_rope":
            u, sv, positions = _reconstruction_fixture(head_dim=64, rank=160)
            positions[0, 0] = -1
            shadowkv.reconstruct_rope(
                u,
                sv,
                positions,
                torch.ones(32, dtype=torch.float32),
                implementation=provider,
            )
        elif operation == "packed_gqa":
            query = torch.zeros((1, 3, 64), dtype=torch.bfloat16)
            keys = torch.zeros((1, 2, 8, 64), dtype=torch.bfloat16)
            shadowkv.packed_gqa(
                query,
                keys,
                torch.zeros_like(keys),
                torch.ones(1, dtype=torch.int32),
                implementation=provider,
            )
        else:
            shadowkv.plan_reuse(
                *_planner_fixture(),
                max_reuse_chunks=3,
                chunk_size=0,
                implementation=provider,
            )
    assert not called


@pytest.mark.parametrize("provider_name", ["fake-cuda", "fake-triton"])
def test_provider_adapter_substitution_preserves_public_contract(provider_name):
    u, sv, positions = _reconstruction_fixture()
    expected = reference.reconstruct(u, sv, positions)
    calls = []

    def fake_provider(*args, **kwargs):
        calls.append(provider_name)
        return reference.reconstruct(*args, **kwargs)

    actual = shadowkv.reconstruct(u, sv, positions, implementation=fake_provider)
    assert torch.equal(actual, expected)
    assert calls == [provider_name]


def test_generic_aot_candidates_have_explicit_qualified_architectures():
    for op, implementation_id in shadowkv.GENERIC_AOT_IMPLEMENTATIONS.items():
        spec = registry.get_implementation(op, implementation_id)
        assert spec.specialization is KernelSpecialization.GENERIC
        assert spec.supported_architectures == frozenset({"sm80", "sm100a"})
        assert "*" not in spec.supported_architectures
        assert len(spec.qualification_references) == 2


def test_import_inventory_is_provider_lazy_and_does_not_initialize_cuda():
    code = (
        "import sys, torch; before=torch.cuda.is_initialized(); "
        "import sglang.kernels.ops; after=torch.cuda.is_initialized(); "
        "dirty=any(name.startswith('sglang.kernels.ops.shadowkv.providers.') "
        "for name in sys.modules) or 'sgl_kernel' in sys.modules; "
        "print(before, after, dirty)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("False False False")


def test_lazy_availability_and_warmup_hooks(monkeypatch):
    import sglang.kernels.spec as spec_module

    events = []

    class Module:
        @staticmethod
        def kernel():
            events.append("load")

        @staticmethod
        def available():
            events.append("probe")
            return True

        @staticmethod
        def warm():
            events.append("warm")

    monkeypatch.setattr(spec_module.importlib, "import_module", lambda _: Module)
    spec = KernelSpec(
        op="shadowkv.test",
        backend=KernelBackend.TRITON,
        target="fake:kernel",
        capabilities=frozenset({Cap.CUDA}),
        implementation_id="shadowkv.test.triton",
        specialization=KernelSpecialization.ARCHITECTURE,
        supported_architectures=frozenset({"sm80"}),
        input_envelope=KernelInputEnvelope(),
        execution=KernelExecutionProperties(graph_compatible=False),
        availability_target="fake:available",
        warmup_target="fake:warm",
    )
    spec.inventory_record()
    assert events == []
    assert spec.provider_available()
    loaded = spec.load()
    assert events == ["probe"]
    loaded()
    spec.warm_up()
    assert events == ["probe", "load", "warm"]


def test_eager_only_triton_candidate_is_excluded_from_graph_selection(monkeypatch):
    import sglang.kernels.selector as selector_module
    from sglang.kernels.registry import KernelRegistry

    candidate = KernelSpec(
        op="shadowkv.test_graph",
        backend=KernelBackend.TRITON,
        target="fake:kernel",
        capabilities=frozenset({Cap.CUDA}),
        implementation_id="shadowkv.test_graph.triton",
        specialization=KernelSpecialization.ARCHITECTURE,
        supported_architectures=frozenset({"sm80"}),
        execution=KernelExecutionProperties(graph_compatible=False),
    )
    local_registry = KernelRegistry()
    local_registry.register(candidate)
    monkeypatch.setattr(selector_module, "registry", local_registry)
    platform = PlatformInfo(
        device_type="cuda", architecture_name="sm80", cuda_arch_major=8
    )
    with pytest.raises(KernelSelectionError) as caught:
        select_kernel_candidate(
            candidate.op,
            platform=platform,
            require_graph=True,
            qualified_implementation_ids=frozenset({candidate.identity}),
        )
    assert "graph-incompatible" in caught.value.rejections[0].reasons


def test_triton_seam_has_no_registered_production_candidate():
    assert providers.__all__ == ["aot", "reference", "triton"]
    assert all(
        spec.backend is not KernelBackend.TRITON
        for op in shadowkv.GENERIC_AOT_IMPLEMENTATIONS
        for spec in registry.get(op)
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
