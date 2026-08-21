import argparse
import contextlib
import io
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

from sglang.srt import runtime_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _parse(*arguments):
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    namespace = parser.parse_args(["--model-path", "dummy", *arguments])
    return ServerArgs.from_cli_args(namespace)


class TestAttentionBackendConfig(CustomTestCase):
    def tearDown(self):
        runtime_context.reset_context()

    def test_parses_inline_json_object(self):
        server_args = _parse(
            "--attention-backend-config",
            '{"mode":"fixture","nested":{"enabled":true}}',
        )

        self.assertEqual(
            server_args.attention_backend_config,
            {"mode": "fixture", "nested": {"enabled": True}},
        )

    def test_parses_path_before_process_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "backend.json"
            path.write_text('{"rank":160,"kernel":"readable"}', encoding="utf-8")
            server_args = _parse("--attention-backend-config", f"@{path}")
            path.unlink()

        transferred = pickle.loads(pickle.dumps(server_args))
        runtime_context.publish(transferred, role="scheduler")

        self.assertEqual(
            runtime_context.get_exec().kernel.attention_backend_config,
            {"rank": 160, "kernel": "readable"},
        )

    def test_rejects_malformed_json(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            _parse("--attention-backend-config", '{"rank":')

    def test_rejects_missing_path(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            _parse("--attention-backend-config", "@/missing/backend-config.json")

    def test_preserves_unknown_keys_for_backend_handoff(self):
        shadowkv_modules_before = {
            name for name in sys.modules if "shadowkv" in name.lower()
        }

        server_args = _parse(
            "--attention-backend-config",
            '{"out_of_tree_option":{"future":1}}',
        )

        shadowkv_modules_after = {
            name for name in sys.modules if "shadowkv" in name.lower()
        }
        self.assertEqual(
            server_args.attention_backend_config,
            {"out_of_tree_option": {"future": 1}},
        )
        self.assertEqual(shadowkv_modules_after, shadowkv_modules_before)

    def test_rejects_non_object_and_duplicate_keys(self):
        for value in ('["not", "an", "object"]', '{"rank":1,"rank":2}'):
            with (
                self.subTest(value=value),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                _parse("--attention-backend-config", value)


if __name__ == "__main__":
    unittest.main()
