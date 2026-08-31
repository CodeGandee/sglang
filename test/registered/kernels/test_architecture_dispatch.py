"""CPU-only tests for architecture-aware kernel inventory and selection."""

import json

import msgspec
import pytest
import sglang.kernels.selector as selector
from sglang.kernels.registry import KernelRegistry
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

_SM80_4 = PlatformInfo(
    device_type="cuda",
    cuda_arch_major=8,
    cuda_arch_minor=0,
    device_index=4,
    architecture_name="sm80",
)
_SM100A_7 = PlatformInfo(
    device_type="cuda",
    cuda_arch_major=10,
    cuda_arch_minor=0,
    device_index=7,
    architecture_name="sm100a",
)
_SM90A = PlatformInfo(
    device_type="cuda",
    cuda_arch_major=9,
    cuda_arch_minor=0,
    device_index=2,
    architecture_name="sm90a",
)
_ENVELOPE = KernelInputEnvelope(
    dtypes=frozenset({"bfloat16"}),
    head_dimensions=frozenset({128}),
    max_batch_size=8,
    features=frozenset({"current-stream"}),
)
_EXECUTION = KernelExecutionProperties(
    deterministic=True,
    graph_compatible=True,
)


def _candidate(
    implementation_id,
    *,
    backend=KernelBackend.AOT,
    specialization=KernelSpecialization.GENERIC,
    architectures=frozenset({"sm80", "sm100a"}),
    target="math:sqrt",
    availability_target=None,
):
    return KernelSpec(
        op="shadowkv.reconstruct",
        backend=backend,
        target=target,
        capabilities=frozenset({Cap.CUDA}),
        operation_revision="v1",
        implementation_id=implementation_id,
        specialization=specialization,
        supported_architectures=architectures,
        input_envelope=_ENVELOPE,
        execution=_EXECUTION,
        availability_target=availability_target,
        qualification_references=("qualification://fixture",),
    )


def _registry(monkeypatch, *specs):
    registry = KernelRegistry()
    for spec in specs:
        registry.register(spec)
    monkeypatch.setattr(selector, "registry", registry)
    return registry


def test_legacy_spec_construction_and_serialization_remain_supported():
    spec = KernelSpec(op="test.legacy", backend=KernelBackend.TORCH, target="math:sqrt")
    assert spec.identity == "torch"
    assert spec.provider is KernelBackend.TORCH
    payload = json.loads(msgspec.json.encode(spec))
    assert payload["operation_revision"] == "legacy-v1"
    assert payload["implementation_id"] is None
    assert payload["specialization"] == "unspecified"


def test_architecture_aware_metadata_round_trips_deterministically():
    spec = _candidate("shadowkv.reconstruct.generic-aot.v1")
    encoded = msgspec.json.encode(spec)
    decoded = msgspec.json.decode(encoded, type=KernelSpec)
    assert decoded == spec
    assert msgspec.json.encode(decoded) == encoded
    inventory = spec.inventory_record()
    assert inventory["implementation_id"] == spec.identity
    assert inventory["provider"] == "aot"


@pytest.mark.parametrize("architectures", [frozenset(), frozenset({"*"})])
def test_generic_metadata_requires_explicit_exact_architectures(architectures):
    with pytest.raises(ValueError):
        _candidate("bad-generic", architectures=architectures)


def test_architecture_specialization_requires_one_exact_architecture():
    with pytest.raises(ValueError):
        _candidate(
            "bad-specialization",
            specialization=KernelSpecialization.ARCHITECTURE,
            architectures=frozenset({"sm80", "sm100a"}),
        )


def test_registry_supports_multiple_implementations_from_one_provider():
    generic = _candidate("generic")
    specialized = _candidate(
        "sm80-specialized",
        specialization=KernelSpecialization.ARCHITECTURE,
        architectures=frozenset({"sm80"}),
        target="math:floor",
    )
    registry = KernelRegistry()
    registry.register(generic)
    registry.register(specialized)
    registry.register(generic)
    assert registry.get_implementation(generic.op, "generic") == generic
    with pytest.raises(ValueError, match="ambiguous"):
        registry.get_backend(generic.op, KernelBackend.AOT)


def test_registry_rejects_duplicate_implementation_identity():
    registry = KernelRegistry()
    registry.register(_candidate("duplicate"))
    with pytest.raises(ValueError, match="Conflicting kernel registration"):
        registry.register(_candidate("duplicate", target="math:floor"))


def test_inventory_is_sorted_and_does_not_probe_or_load_provider(monkeypatch):
    import sglang.kernels.spec as spec_module

    calls = []
    monkeypatch.setattr(
        spec_module.importlib,
        "import_module",
        lambda name: calls.append(name) or pytest.fail("provider imported"),
    )
    registry = KernelRegistry()
    registry.register(
        _candidate(
            "z-generic",
            target="missing.provider:kernel",
            availability_target="missing.provider:available",
        )
    )
    records = registry.inventory()
    assert records[0]["implementation_id"] == "z-generic"
    assert calls == []


def test_auto_prefers_qualified_exact_specialization(monkeypatch):
    generic = _candidate("generic")
    specialized = _candidate(
        "sm80-specialized",
        specialization=KernelSpecialization.ARCHITECTURE,
        architectures=frozenset({"sm80"}),
        target="math:floor",
    )
    _registry(monkeypatch, generic, specialized)
    result = select_kernel_candidate(
        generic.op,
        platform=_SM80_4,
        envelope=_ENVELOPE,
        packaged_implementation_ids=frozenset({"generic", "sm80-specialized"}),
        qualified_implementation_ids=frozenset({"generic", "sm80-specialized"}),
    )
    assert result.selected.identity == "sm80-specialized"
    assert result.platform.device_index == 4
    assert result.fallback_reason is None


def test_auto_falls_back_to_qualified_generic_with_reason(monkeypatch):
    generic = _candidate("generic")
    specialized = _candidate(
        "sm80-specialized",
        specialization=KernelSpecialization.ARCHITECTURE,
        architectures=frozenset({"sm80"}),
        target="math:floor",
    )
    _registry(monkeypatch, specialized, generic)
    result = select_kernel_candidate(
        generic.op,
        platform=_SM100A_7,
        envelope=_ENVELOPE,
        qualified_implementation_ids=frozenset({"generic", "sm80-specialized"}),
    )
    assert result.selected.identity == "generic"
    assert result.platform.device_index == 7
    assert "sm100a" in result.fallback_reason
    rejected = {item.implementation_id: item.reasons for item in result.rejections}
    assert rejected["sm80-specialized"] == ("architecture:sm100a",)


def test_available_but_unqualified_specialization_cannot_displace_generic(
    monkeypatch,
):
    generic = _candidate("generic")
    specialized = _candidate(
        "sm80-specialized",
        specialization=KernelSpecialization.ARCHITECTURE,
        architectures=frozenset({"sm80"}),
        target="math:floor",
    )
    _registry(monkeypatch, generic, specialized)
    result = select_kernel_candidate(
        generic.op,
        platform=_SM80_4,
        envelope=_ENVELOPE,
        qualified_implementation_ids=frozenset({"generic"}),
    )
    assert result.selected.identity == "generic"
    assert result.rejections[-1].reasons == ("unqualified",)


def test_unpackaged_optional_specialization_falls_back_before_provider_load(
    monkeypatch,
):
    generic = _candidate("generic")
    specialized = _candidate(
        "sm80-optional",
        specialization=KernelSpecialization.ARCHITECTURE,
        architectures=frozenset({"sm80"}),
        target="missing.optional:kernel",
    )
    _registry(monkeypatch, specialized, generic)
    result = select_kernel_candidate(
        generic.op,
        platform=_SM80_4,
        envelope=_ENVELOPE,
        packaged_implementation_ids=frozenset({"generic"}),
        qualified_implementation_ids=frozenset({"generic", "sm80-optional"}),
    )
    assert result.selected.identity == "generic"
    assert result.rejections[0].reasons == ("not-packaged",)


def test_declared_provider_probe_failure_does_not_fall_back(monkeypatch):
    import sglang.kernels.spec as spec_module

    generic = _candidate("generic")
    specialized = _candidate(
        "sm80-declared",
        specialization=KernelSpecialization.ARCHITECTURE,
        architectures=frozenset({"sm80"}),
        target="broken.provider:kernel",
        availability_target="broken.provider:available",
    )
    _registry(monkeypatch, specialized, generic)

    def fail_import(name):
        raise ImportError(f"injected failure for {name}")

    monkeypatch.setattr(spec_module.importlib, "import_module", fail_import)
    with pytest.raises(KernelSelectionError) as caught:
        select_kernel_candidate(
            generic.op,
            platform=_SM80_4,
            envelope=_ENVELOPE,
            packaged_implementation_ids=frozenset({"generic", "sm80-declared"}),
            qualified_implementation_ids=frozenset({"generic", "sm80-declared"}),
            probe_provider=True,
        )
    assert caught.value.failure_class == "provider-probe-failure"


def test_strict_aot_provider_constraint_never_selects_triton(monkeypatch):
    generic = _candidate("generic")
    triton_specialized = _candidate(
        "sm80-triton",
        backend=KernelBackend.TRITON,
        specialization=KernelSpecialization.ARCHITECTURE,
        architectures=frozenset({"sm80"}),
        target="math:floor",
    )
    _registry(monkeypatch, triton_specialized, generic)
    result = select_kernel_candidate(
        generic.op,
        provider=KernelBackend.AOT,
        platform=_SM80_4,
        envelope=_ENVELOPE,
        packaged_implementation_ids=frozenset({"generic", "sm80-triton"}),
        qualified_implementation_ids=frozenset({"generic", "sm80-triton"}),
    )
    assert result.selected.identity == "generic"
    assert result.rejections[0].reasons == ("provider-policy:aot",)


def test_exact_generic_and_reference_policies_are_strict(monkeypatch):
    generic = _candidate("generic")
    reference = _candidate(
        "reference",
        backend=KernelBackend.TORCH,
        specialization=KernelSpecialization.REFERENCE,
        architectures=frozenset(),
        target="math:ceil",
    )
    _registry(monkeypatch, reference, generic)
    exact = select_kernel_candidate(
        generic.op,
        policy=KernelSelectionPolicy.EXACT,
        implementation_id="generic",
        platform=_SM80_4,
        envelope=_ENVELOPE,
        qualified_implementation_ids=frozenset({"generic", "reference"}),
    )
    assert exact.selected.identity == "generic"
    forced_reference = select_kernel_candidate(
        generic.op,
        policy=KernelSelectionPolicy.REFERENCE,
        platform=_SM80_4,
        envelope=_ENVELOPE,
        qualified_implementation_ids=frozenset({"generic", "reference"}),
    )
    assert forced_reference.selected.identity == "reference"
    with pytest.raises(KernelSelectionError) as caught:
        select_kernel_candidate(
            generic.op,
            policy=KernelSelectionPolicy.EXACT,
            implementation_id="absent",
            platform=_SM80_4,
            envelope=_ENVELOPE,
        )
    assert caught.value.failure_class == "explicit-request-unavailable"


@pytest.mark.parametrize("reverse", [False, True])
def test_equal_preference_fails_independent_of_registration_order(monkeypatch, reverse):
    first = _candidate("generic-a")
    second = _candidate("generic-b", target="math:floor")
    specs = [first, second]
    if reverse:
        specs.reverse()
    _registry(monkeypatch, *specs)
    with pytest.raises(KernelSelectionError) as caught:
        select_kernel_candidate(
            first.op,
            platform=_SM80_4,
            envelope=_ENVELOPE,
            qualified_implementation_ids=frozenset({"generic-a", "generic-b"}),
        )
    assert caught.value.failure_class == "ambiguous-preference"


def test_preference_list_breaks_same_tier_tie(monkeypatch):
    first = _candidate("generic-a")
    second = _candidate("generic-b", target="math:floor")
    _registry(monkeypatch, first, second)
    result = select_kernel_candidate(
        first.op,
        platform=_SM80_4,
        envelope=_ENVELOPE,
        qualified_implementation_ids=frozenset({"generic-a", "generic-b"}),
        preferred_implementation_ids=("generic-b", "generic-a"),
    )
    assert result.selected.identity == "generic-b"


def test_simulated_multi_device_resolution_uses_each_operation_device(monkeypatch):
    generic = _candidate("generic")
    sm80 = _candidate(
        "sm80-specialized",
        specialization=KernelSpecialization.ARCHITECTURE,
        architectures=frozenset({"sm80"}),
        target="math:floor",
    )
    sm100a = _candidate(
        "sm100a-specialized",
        specialization=KernelSpecialization.ARCHITECTURE,
        architectures=frozenset({"sm100a"}),
        target="math:ceil",
    )
    _registry(monkeypatch, sm100a, generic, sm80)
    qualified = frozenset({"generic", "sm80-specialized", "sm100a-specialized"})
    result_4 = select_kernel_candidate(
        generic.op,
        platform=_SM80_4,
        envelope=_ENVELOPE,
        qualified_implementation_ids=qualified,
    )
    result_7 = select_kernel_candidate(
        generic.op,
        platform=_SM100A_7,
        envelope=_ENVELOPE,
        qualified_implementation_ids=qualified,
    )
    assert (result_4.platform.device_index, result_4.selected.identity) == (
        4,
        "sm80-specialized",
    )
    assert (result_7.platform.device_index, result_7.selected.identity) == (
        7,
        "sm100a-specialized",
    )


def test_unsupported_architecture_and_envelope_report_rejections(monkeypatch):
    generic = _candidate("generic")
    _registry(monkeypatch, generic)
    unsupported = KernelInputEnvelope(dtypes=frozenset({"float32"}))
    with pytest.raises(KernelSelectionError) as caught:
        select_kernel_candidate(
            generic.op,
            platform=_SM90A,
            envelope=unsupported,
            qualified_implementation_ids=frozenset({"generic"}),
        )
    assert caught.value.failure_class == "no-qualified-candidate"
    reasons = caught.value.rejections[0].reasons
    assert "architecture:sm90a" in reasons
    assert "input-envelope" in reasons


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
