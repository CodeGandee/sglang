from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sglang.srt.layers.attention import attention_registry
from sglang.srt.layers.attention.child_backend_factory import (
    CHILD_BACKEND_FACTORY_SCHEMA_VERSION,
    AttentionPhaseRole,
    AttentionWorkspacePolicy,
    FullAttentionCacheAccessContract,
    FullAttentionChildBackendRequest,
    construct_full_attention_child_backend,
    get_child_backend_construction_context,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeBackend:
    def __init__(self, name: str):
        self.name = name
        self.closed = 0

    def close(self):
        self.closed += 1

    def forward(self, query, key, value):
        return tuple(q + k + v for q, k, v in zip(query, key, value, strict=True))


def _runner():
    request_pool = object()
    token_pool = object()
    allocator = object()
    return SimpleNamespace(
        req_to_token_pool=request_pool,
        token_to_kv_pool=token_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
        kv_cache_dtype="bf16",
        init_new_workspace="outer-workspace",
        prefill_attention_backend_str="shadowkv",
        decode_attention_backend_str="shadowkv",
    )


def _request(runner, *roles, workspace=AttentionWorkspacePolicy.SHARED):
    return FullAttentionChildBackendRequest(
        schema_version=CHILD_BACKEND_FACTORY_SCHEMA_VERSION,
        backend_name="fixture",
        effective_backend_identity="fixture.provider.v1",
        phase_roles=tuple(roles),
        workspace_policy=workspace,
        cache_access=FullAttentionCacheAccessContract(
            schema_version=CHILD_BACKEND_FACTORY_SCHEMA_VERSION,
            contract_id="fixture-cache-v1",
            request_pool=runner.req_to_token_pool,
            token_pool=runner.token_to_kv_pool,
            allocator=runner.token_to_kv_pool_allocator,
            page_size=runner.page_size,
            kv_cache_dtype=runner.kv_cache_dtype,
        ),
    )


def test_constructs_single_role_with_effective_identity_and_restores_outer_runner():
    runner = _runner()
    observed = {}

    def create(model_runner):
        observed["workspace"] = model_runner.init_new_workspace
        observed["prefill"] = model_runner.prefill_attention_backend_str
        observed["decode"] = model_runner.decode_attention_backend_str
        context = get_child_backend_construction_context(model_runner)
        assert context is not None
        observed["request"] = context.request
        return _FakeBackend("fixture")

    with patch.dict(attention_registry.ATTENTION_BACKENDS, {"fixture": create}):
        child = construct_full_attention_child_backend(
            model_runner=runner,
            request=_request(runner, AttentionPhaseRole.PREFILL),
        )

    assert observed == {
        "workspace": False,
        "prefill": "fixture",
        "decode": "fixture",
        "request": child.request,
    }
    assert child.effective_backend_identity == "fixture.provider.v1"
    assert child.backend.prefill_attention_backend_str == "fixture"
    assert child.backend.decode_attention_backend_str is None
    assert runner.init_new_workspace == "outer-workspace"
    assert runner.prefill_attention_backend_str == "shadowkv"
    assert runner.decode_attention_backend_str == "shadowkv"


def test_constructs_one_child_for_multiple_roles_and_dedicated_workspace():
    runner = _runner()
    calls = []

    def create(model_runner):
        calls.append(model_runner.init_new_workspace)
        return _FakeBackend("fixture")

    request = _request(
        runner,
        AttentionPhaseRole.PREFILL,
        AttentionPhaseRole.DECODE,
        AttentionPhaseRole.TARGET_VERIFY,
        workspace=AttentionWorkspacePolicy.DEDICATED,
    )
    with patch.dict(attention_registry.ATTENTION_BACKENDS, {"fixture": create}):
        child = construct_full_attention_child_backend(
            model_runner=runner, request=request
        )

    assert calls == [True]
    assert child.phase_roles == request.phase_roles
    assert child.backend.prefill_attention_backend_str == "fixture"
    assert child.backend.decode_attention_backend_str == "fixture"


def test_invalid_name_fails_without_mutating_runner():
    runner = _runner()
    before = vars(runner).copy()

    with (
        patch.dict(attention_registry.ATTENTION_BACKENDS, {}, clear=True),
        pytest.raises(ValueError, match="Invalid attention backend"),
    ):
        construct_full_attention_child_backend(
            model_runner=runner,
            request=_request(runner, AttentionPhaseRole.DECODE),
        )

    assert vars(runner) == before


def test_initialization_failure_rolls_back_callbacks_and_runner_mutations():
    runner = _runner()
    before = vars(runner).copy()
    events = []

    def create(model_runner):
        context = get_child_backend_construction_context(model_runner)
        assert context is not None
        context.register_rollback(lambda: events.append("rollback"))
        context.register_cleanup(lambda: events.append("unexpected-cleanup"))
        model_runner.provider_stream = object()
        raise RuntimeError("fixture initialization failed")

    with (
        patch.dict(attention_registry.ATTENTION_BACKENDS, {"fixture": create}),
        pytest.raises(RuntimeError, match="fixture initialization failed"),
    ):
        construct_full_attention_child_backend(
            model_runner=runner,
            request=_request(runner, AttentionPhaseRole.DECODE),
        )

    assert events == ["rollback"]
    assert vars(runner) == before


def test_close_releases_once_and_restores_provider_runner_state():
    runner = _runner()
    events = []

    def create(model_runner):
        context = get_child_backend_construction_context(model_runner)
        assert context is not None
        context.register_cleanup(lambda: events.append("cleanup"))
        model_runner.provider_workspace = object()
        return _FakeBackend("fixture")

    with patch.dict(attention_registry.ATTENTION_BACKENDS, {"fixture": create}):
        child = construct_full_attention_child_backend(
            model_runner=runner,
            request=_request(runner, AttentionPhaseRole.EXACT_FALLBACK),
        )

    assert hasattr(runner, "provider_workspace")
    child.close()
    child.close()
    assert child.backend.closed == 1
    assert events == ["cleanup"]
    assert not hasattr(runner, "provider_workspace")


def test_cache_contract_rejects_a_different_pool_before_provider_lookup():
    runner = _runner()
    request = _request(runner, AttentionPhaseRole.PREFILL)
    request = FullAttentionChildBackendRequest(
        schema_version=request.schema_version,
        backend_name=request.backend_name,
        effective_backend_identity=request.effective_backend_identity,
        phase_roles=request.phase_roles,
        workspace_policy=request.workspace_policy,
        cache_access=FullAttentionCacheAccessContract(
            schema_version=request.cache_access.schema_version,
            contract_id=request.cache_access.contract_id,
            request_pool=request.cache_access.request_pool,
            token_pool=object(),
            allocator=request.cache_access.allocator,
            page_size=request.cache_access.page_size,
            kv_cache_dtype=request.cache_access.kv_cache_dtype,
        ),
    )

    with pytest.raises(ValueError, match="token_pool"):
        construct_full_attention_child_backend(model_runner=runner, request=request)


def test_inventory_and_selected_construction_leave_inactive_provider_uninitialized():
    runner = _runner()
    events = []

    def active(_model_runner):
        events.append("active-constructed")
        return _FakeBackend("active")

    def inactive(_model_runner):
        events.extend(
            (
                "inactive-state",
                "inactive-workspace",
                "inactive-stream",
                "inactive-graph",
                "inactive-compiled-kernel",
            )
        )
        return _FakeBackend("inactive")

    with patch.dict(
        attention_registry.ATTENTION_BACKENDS,
        {"fixture": active, "inactive": inactive},
        clear=True,
    ):
        assert attention_registry.registered_attention_backend_names() == (
            "fixture",
            "inactive",
        )
        child = construct_full_attention_child_backend(
            model_runner=runner,
            request=_request(runner, AttentionPhaseRole.PREFILL),
        )

    assert events == ["active-constructed"]
    child.close()


def test_repeated_worker_initialization_owns_and_cleans_resources_independently():
    runners = (_runner(), _runner())
    cleanup = []

    def create(model_runner):
        context = get_child_backend_construction_context(model_runner)
        assert context is not None
        worker_id = id(model_runner)
        context.register_cleanup(lambda: cleanup.append(worker_id))
        model_runner.provider_workspace = object()
        return _FakeBackend(str(worker_id))

    with patch.dict(attention_registry.ATTENTION_BACKENDS, {"fixture": create}):
        children = tuple(
            construct_full_attention_child_backend(
                model_runner=runner,
                request=_request(runner, AttentionPhaseRole.DECODE),
            )
            for runner in runners
        )

    assert children[0].backend is not children[1].backend
    assert runners[0].provider_workspace is not runners[1].provider_workspace
    children[0].close()
    assert not hasattr(runners[0], "provider_workspace")
    assert hasattr(runners[1], "provider_workspace")
    children[1].close()
    assert cleanup == [id(runners[0]), id(runners[1])]
    assert not hasattr(runners[1], "provider_workspace")


def test_child_seam_preserves_incumbent_outputs_allocation_and_launch_identity():
    legacy_runner = _runner()
    child_runner = _runner()
    launches = []
    allocations = []

    def create(model_runner):
        allocations.append(id(model_runner))
        launches.append(
            (
                model_runner.prefill_attention_backend_str,
                model_runner.decode_attention_backend_str,
            )
        )
        return _FakeBackend("triton")

    query = (1.0, -2.0, 3.5)
    key = (0.5, 4.0, -1.5)
    value = (2.0, 1.0, 0.25)
    with patch.dict(attention_registry.ATTENTION_BACKENDS, {"triton": create}):
        incumbent = create(legacy_runner)
        request = FullAttentionChildBackendRequest(
            schema_version=CHILD_BACKEND_FACTORY_SCHEMA_VERSION,
            backend_name="triton",
            effective_backend_identity="sglang.triton.full-attention.v1",
            phase_roles=(
                AttentionPhaseRole.PREFILL,
                AttentionPhaseRole.DECODE,
                AttentionPhaseRole.EXACT_FALLBACK,
            ),
            workspace_policy=AttentionWorkspacePolicy.SHARED,
            cache_access=FullAttentionCacheAccessContract(
                schema_version=CHILD_BACKEND_FACTORY_SCHEMA_VERSION,
                contract_id="sglang-native-dense-cache-v1",
                request_pool=child_runner.req_to_token_pool,
                token_pool=child_runner.token_to_kv_pool,
                allocator=child_runner.token_to_kv_pool_allocator,
                page_size=child_runner.page_size,
                kv_cache_dtype=child_runner.kv_cache_dtype,
            ),
        )
        child = construct_full_attention_child_backend(
            model_runner=child_runner,
            request=request,
        )

    assert child.backend.forward(query, key, value) == incumbent.forward(
        query, key, value
    )
    assert allocations == [id(legacy_runner), id(child_runner)]
    assert launches == [("shadowkv", "shadowkv"), ("triton", "triton")]
    assert child.effective_backend_identity == "sglang.triton.full-attention.v1"
    assert child_runner.prefill_attention_backend_str == "shadowkv"
    assert child_runner.decode_attention_backend_str == "shadowkv"
    child.close()
