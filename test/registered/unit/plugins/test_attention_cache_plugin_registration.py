"""Process-level tests for attention and cache-provider plugin registration."""

import argparse
import multiprocessing
import traceback
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.plugins as plugins_module
from sglang.srt.layers.attention.attention_registry import (
    ATTENTION_BACKENDS,
    register_attention_backend,
)
from sglang.srt.mem_cache.cache_provider_registry import (
    ATTENTION_BACKEND_CACHE_PROVIDERS,
    AttentionBackendCacheResult,
    register_attention_backend_cache_provider,
    select_attention_backend_cache_provider,
)
from sglang.srt.server_args import (
    ATTENTION_BACKEND_CHOICES,
    ServerArgs,
    add_attention_backend_choices,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_PLUGIN_NAME = "unit_plugin_attention"
_registration_count = 0
_attention_factory_count = 0
_cache_factory_count = 0


def _create_attention_backend(runner):
    global _attention_factory_count
    _attention_factory_count += 1
    return SimpleNamespace(kind="attention", runner=runner)


def _create_cache_provider(configurator, *, pre_model_load_memory):
    global _cache_factory_count
    _cache_factory_count += 1
    return AttentionBackendCacheResult(
        max_total_num_tokens=16,
        max_running_requests=2,
        full_max_total_num_tokens=None,
        swa_max_total_num_tokens=None,
        req_to_token_pool=object(),
        token_to_kv_pool=object(),
        token_to_kv_pool_allocator=object(),
        memory_pool_config=(configurator, pre_model_load_memory),
    )


def _register_fixture_plugin():
    global _registration_count
    _registration_count += 1
    add_attention_backend_choices([_PLUGIN_NAME])
    register_attention_backend(_PLUGIN_NAME)(_create_attention_backend)
    register_attention_backend_cache_provider(_PLUGIN_NAME)(_create_cache_provider)


class _EntryPoint:
    name = _PLUGIN_NAME
    value = f"{__name__}:_register_fixture_plugin"
    dist = SimpleNamespace(name="unit-plugin-dist")

    @staticmethod
    def load():
        return _register_fixture_plugin


def _load_fixture_plugin_twice():
    plugins_module._plugins_loaded = False
    with (
        patch.object(plugins_module, "entry_points", return_value=[_EntryPoint()]),
        patch.object(plugins_module, "envs") as envs,
        patch.object(plugins_module.HookRegistry, "apply_hooks") as apply_hooks,
    ):
        envs.SGLANG_PLATFORM.get.return_value = ""
        envs.SGLANG_PLUGINS.get.return_value = ""
        plugins_module.load_plugins()
        plugins_module.load_plugins()
    return apply_hooks.call_count


def _probe_registered_factories():
    runner = object()
    backend = ATTENTION_BACKENDS[_PLUGIN_NAME](runner)
    selected = select_attention_backend_cache_provider((_PLUGIN_NAME,))
    if selected is None:
        raise AssertionError("cache provider was not selected")
    cache_result = selected.factory(object(), pre_model_load_memory=37)
    return {
        "backend_kind": backend.kind,
        "backend_runner_matches": backend.runner is runner,
        "cache_tokens": cache_result.max_total_num_tokens,
        "choice_count": ATTENTION_BACKEND_CHOICES.count(_PLUGIN_NAME),
        "registration_count": _registration_count,
        "attention_factory_count": _attention_factory_count,
        "cache_factory_count": _cache_factory_count,
    }


def _spawn_probe(queue):
    try:
        apply_count = _load_fixture_plugin_twice()
        queue.put({"ok": True, "apply_count": apply_count, **_probe_registered_factories()})
    except Exception:
        queue.put({"ok": False, "traceback": traceback.format_exc()})
        raise


def _reset_fixture_state():
    global _registration_count, _attention_factory_count, _cache_factory_count
    _registration_count = 0
    _attention_factory_count = 0
    _cache_factory_count = 0
    ATTENTION_BACKENDS.pop(_PLUGIN_NAME, None)
    ATTENTION_BACKEND_CACHE_PROVIDERS.pop(_PLUGIN_NAME, None)
    ATTENTION_BACKEND_CHOICES[:] = [
        choice for choice in ATTENTION_BACKEND_CHOICES if choice != _PLUGIN_NAME
    ]
    plugins_module._plugins_loaded = False


class TestAttentionCachePluginRegistration(CustomTestCase):
    def setUp(self):
        _reset_fixture_state()

    def tearDown(self):
        _reset_fixture_state()

    def test_choice_and_factories_are_lazy_and_loader_is_idempotent(self):
        apply_count = _load_fixture_plugin_twice()

        self.assertEqual(apply_count, 1)
        self.assertEqual(_registration_count, 1)
        self.assertEqual(_attention_factory_count, 0)
        self.assertEqual(_cache_factory_count, 0)
        self.assertEqual(ATTENTION_BACKEND_CHOICES.count(_PLUGIN_NAME), 1)

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        namespace = parser.parse_args(
            ["--model-path", "dummy", "--attention-backend", _PLUGIN_NAME]
        )
        self.assertEqual(namespace.attention_backend, _PLUGIN_NAME)

        probe = _probe_registered_factories()
        self.assertEqual(probe["backend_kind"], "attention")
        self.assertTrue(probe["backend_runner_matches"])
        self.assertEqual(probe["cache_tokens"], 16)
        self.assertEqual(probe["attention_factory_count"], 1)
        self.assertEqual(probe["cache_factory_count"], 1)

    def test_spawned_process_loads_and_constructs_once(self):
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        process = context.Process(target=_spawn_probe, args=(queue,))
        process.start()
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            self.fail("spawned plugin probe did not finish")

        result = queue.get(timeout=5)
        self.assertEqual(process.exitcode, 0, result)
        self.assertTrue(result["ok"], result.get("traceback"))
        self.assertEqual(result["apply_count"], 1)
        self.assertEqual(result["registration_count"], 1)
        self.assertEqual(result["choice_count"], 1)
        self.assertEqual(result["attention_factory_count"], 1)
        self.assertEqual(result["cache_factory_count"], 1)

    def test_conflicting_attention_factory_is_rejected(self):
        register_attention_backend(_PLUGIN_NAME)(_create_attention_backend)
        register_attention_backend(_PLUGIN_NAME)(_create_attention_backend)

        with self.assertRaisesRegex(ValueError, "already registered"):
            register_attention_backend(_PLUGIN_NAME)(lambda runner: runner)


if __name__ == "__main__":
    unittest.main()
