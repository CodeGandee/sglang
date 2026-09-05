import argparse
import contextlib
import io
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.srt import runtime_context
from sglang.srt.arg_groups.attention_backend_config import (
    ATTENTION_BACKEND_CONFIG_OWNERS,
    register_attention_backend_config_owner,
    validate_attention_backend_config,
)
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

    def test_selected_owner_validates_during_server_args_resolution(self):
        seen = []

        with patch.dict(ATTENTION_BACKEND_CONFIG_OWNERS, {}, clear=True):

            @register_attention_backend_config_owner("flashinfer")
            def validate(value):
                seen.append(value)

            server_args = _parse(
                "--attention-backend",
                "flashinfer",
                "--attention-backend-config",
                '{"mode":"fixture"}',
            )

        self.assertEqual(seen, [{"mode": "fixture"}])
        self.assertEqual(server_args.get_attention_backends(), ("flashinfer",) * 2)

    def test_rejects_inactive_owner_signature_during_server_args_resolution(self):
        with patch.dict(ATTENTION_BACKEND_CONFIG_OWNERS, {}, clear=True):
            register_attention_backend_config_owner(
                "unit-shadowkv",
                inactive_signature_keys={"sparse_budget"},
            )(lambda value: None)

            with self.assertRaisesRegex(
                ValueError,
                "keys \\[sparse_budget\\] require backend 'unit-shadowkv'",
            ):
                _parse(
                    "--attention-backend",
                    "flashinfer",
                    "--attention-backend-config",
                    '{"mode":"host-sync","sparse_budget":2048}',
                )

    def test_preserves_unclaimed_config_for_an_unregistered_owner(self):
        with patch.dict(ATTENTION_BACKEND_CONFIG_OWNERS, {}, clear=True):
            register_attention_backend_config_owner(
                "unit-shadowkv",
                inactive_signature_keys={"sparse_budget"},
            )(lambda value: None)

            server_args = _parse(
                "--attention-backend",
                "flashinfer",
                "--attention-backend-config",
                '{"mode":"another-plugin","future":true}',
            )

        self.assertEqual(
            server_args.attention_backend_config,
            {"mode": "another-plugin", "future": True},
        )

    def test_rejects_split_registered_config_owners(self):
        with patch.dict(ATTENTION_BACKEND_CONFIG_OWNERS, {}, clear=True):
            register_attention_backend_config_owner("prefill-owner")(lambda value: None)
            register_attention_backend_config_owner("decode-owner")(lambda value: None)

            with self.assertRaisesRegex(ValueError, "incompatible config owners"):
                validate_attention_backend_config(
                    {},
                    ("prefill-owner", "decode-owner"),
                )

    def test_rejects_conflicting_owner_registration(self):
        first = lambda value: None
        second = lambda value: None
        with patch.dict(ATTENTION_BACKEND_CONFIG_OWNERS, {}, clear=True):
            register_attention_backend_config_owner("fixture")(first)
            register_attention_backend_config_owner("fixture")(first)
            with self.assertRaisesRegex(ValueError, "already registered"):
                register_attention_backend_config_owner("fixture")(second)


if __name__ == "__main__":
    unittest.main()
