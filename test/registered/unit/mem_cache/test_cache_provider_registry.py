import types
import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.mem_cache.cache_provider_registry import (
    ATTENTION_BACKEND_CACHE_PROVIDERS,
    AttentionBackendCacheResult,
    register_attention_backend_cache_provider,
)
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _SpecAlgorithm:
    @staticmethod
    def is_none():
        return True


class _ConfiguratorFixture:
    def __init__(self, backend_names):
        self.server_args = types.SimpleNamespace(
            get_attention_backends=lambda: backend_names
        )
        self.spec_algorithm = _SpecAlgorithm()
        self.is_draft_worker = False
        self.memory_pool_config = None
        self._native_memory_pool_config_preview = None
        self._native_memory_pool_config_preview_input = None
        self.req_to_token_pool = None
        self.token_to_kv_pool_allocator = None
        self.device = "cuda"
        self.gpu_id = 0
        self.builtin_config = object()
        self.builtin_sizes = types.SimpleNamespace(
            max_total_num_tokens=128,
            max_running_requests=4,
            full_max_total_num_tokens=None,
            swa_max_total_num_tokens=None,
        )
        self.builtin_pools = types.SimpleNamespace(
            req_to_token_pool=object(),
            token_to_kv_pool=object(),
            token_to_kv_pool_allocator=object(),
            unified_memory_pool=None,
        )
        self.builtin_calls = []

    def _resolve_memory_pool_config(self, pre_model_load_memory):
        self.builtin_calls.append(("resolve", pre_model_load_memory))
        return self.builtin_config

    def _configure_attention_backend_cache_provider(self, **kwargs):
        return KVCacheConfigurator._configure_attention_backend_cache_provider(
            self, **kwargs
        )

    def preview_native_memory_pool_config(self, pre_model_load_memory):
        return KVCacheConfigurator.preview_native_memory_pool_config(
            self, pre_model_load_memory
        )

    def _derive_pool_sizes(self, *, config):
        self.builtin_calls.append(("derive", config))
        return self.builtin_sizes

    def _init_pools(self, **kwargs):
        self.builtin_calls.append(("init", kwargs))
        return self.builtin_pools


def _provider_result():
    return AttentionBackendCacheResult(
        max_total_num_tokens=64,
        max_running_requests=2,
        full_max_total_num_tokens=None,
        swa_max_total_num_tokens=None,
        req_to_token_pool=object(),
        token_to_kv_pool=object(),
        token_to_kv_pool_allocator=object(),
        memory_pool_config=object(),
    )


class TestCacheProviderRegistry(CustomTestCase):
    def setUp(self):
        self.registry = patch.dict(ATTENTION_BACKEND_CACHE_PROVIDERS, clear=True)
        self.registry.start()

    def tearDown(self):
        self.registry.stop()

    @patch(
        "sglang.srt.mem_cache.kv_cache_configurator.get_available_gpu_memory",
        return_value=100.0,
    )
    def test_registered_provider_is_lazy_and_bypasses_builtin_allocation(self, _):
        calls = []
        expected = _provider_result()

        @register_attention_backend_cache_provider("fixture")
        def provider(configurator, *, pre_model_load_memory):
            calls.append((configurator, pre_model_load_memory))
            return expected

        fixture = _ConfiguratorFixture(("fixture", "fixture"))
        self.assertEqual(calls, [])

        result = KVCacheConfigurator.configure(fixture, pre_model_load_memory=123)

        self.assertEqual(calls, [(fixture, 123)])
        self.assertEqual(fixture.builtin_calls, [])
        self.assertIs(result.token_to_kv_pool, expected.token_to_kv_pool)
        self.assertEqual(result.max_total_num_tokens, 64)

    @patch(
        "sglang.srt.mem_cache.kv_cache_configurator.get_available_gpu_memory",
        return_value=100.0,
    )
    def test_dense_backend_preserves_builtin_pool_chain(self, _):
        calls = []

        @register_attention_backend_cache_provider("fixture")
        def provider(*args, **kwargs):
            calls.append((args, kwargs))
            return _provider_result()

        fixture = _ConfiguratorFixture(("triton", "triton"))

        result = KVCacheConfigurator.configure(fixture, pre_model_load_memory=321)

        self.assertEqual(calls, [])
        self.assertEqual(
            [name for name, _ in fixture.builtin_calls],
            ["resolve", "derive", "init"],
        )
        self.assertIs(
            result.token_to_kv_pool,
            fixture.builtin_pools.token_to_kv_pool,
        )
        self.assertIs(result.memory_pool_config, fixture.builtin_config)

    @patch(
        "sglang.srt.mem_cache.kv_cache_configurator.get_available_gpu_memory",
        return_value=100.0,
    )
    def test_provider_lifecycle_is_bound_to_selected_request_pool(self, _):
        callbacks = object()
        request_pool = MagicMock()
        expected = AttentionBackendCacheResult(
            max_total_num_tokens=64,
            max_running_requests=2,
            full_max_total_num_tokens=None,
            swa_max_total_num_tokens=None,
            req_to_token_pool=request_pool,
            token_to_kv_pool=object(),
            token_to_kv_pool_allocator=object(),
            memory_pool_config=object(),
            cache_lifecycle=callbacks,
        )

        @register_attention_backend_cache_provider("fixture")
        def provider(*args, **kwargs):
            return expected

        fixture = _ConfiguratorFixture(("fixture", "fixture"))

        KVCacheConfigurator.configure(fixture, pre_model_load_memory=123)

        request_pool.bind_cache_lifecycle.assert_called_once_with(callbacks)

    @patch(
        "sglang.srt.mem_cache.kv_cache_configurator.get_available_gpu_memory",
        return_value=100.0,
    )
    def test_provider_can_decline_to_builtin_chain(self, _):
        @register_attention_backend_cache_provider("fixture")
        def provider(*args, **kwargs):
            return None

        fixture = _ConfiguratorFixture(("fixture", "fixture"))

        result = KVCacheConfigurator.configure(fixture, pre_model_load_memory=10)

        self.assertEqual(
            [name for name, _ in fixture.builtin_calls],
            ["resolve", "derive", "init"],
        )
        self.assertIs(
            result.token_to_kv_pool,
            fixture.builtin_pools.token_to_kv_pool,
        )

    @patch(
        "sglang.srt.mem_cache.kv_cache_configurator.get_available_gpu_memory",
        return_value=100.0,
    )
    def test_provider_preview_is_reused_by_builtin_allocation(self, _):
        previews = []

        @register_attention_backend_cache_provider("fixture")
        def provider(configurator, *, pre_model_load_memory):
            previews.append(
                configurator.preview_native_memory_pool_config(
                    pre_model_load_memory
                )
            )
            return None

        fixture = _ConfiguratorFixture(("fixture",))
        result = KVCacheConfigurator.configure(fixture, pre_model_load_memory=10)

        self.assertEqual(
            [name for name, _ in fixture.builtin_calls],
            ["resolve", "derive", "init"],
        )
        self.assertIs(previews[0], fixture.builtin_config)
        self.assertIs(result.memory_pool_config, previews[0])

    def test_native_pool_preview_rejects_changed_memory_input(self):
        fixture = _ConfiguratorFixture(("triton",))
        first = fixture.preview_native_memory_pool_config(10)

        self.assertIs(first, fixture.preview_native_memory_pool_config(10))
        with self.assertRaisesRegex(RuntimeError, "preview input changed"):
            fixture.preview_native_memory_pool_config(11)

    def test_conflicting_split_backend_providers_are_rejected(self):
        register_attention_backend_cache_provider("prefill")(
            lambda *args, **kwargs: None
        )
        register_attention_backend_cache_provider("decode")(
            lambda *args, **kwargs: None
        )
        fixture = _ConfiguratorFixture(("prefill", "decode"))

        with self.assertRaisesRegex(ValueError, "incompatible cache providers"):
            KVCacheConfigurator.configure(fixture, pre_model_load_memory=10)


if __name__ == "__main__":
    unittest.main()
