import unittest
from types import SimpleNamespace

import torch

from sglang.srt.configs.mamba_utils import mamba2_state_dtype
from sglang.srt.environ import envs


class TestMambaStateDType(unittest.TestCase):
    def setUp(self):
        envs.SGLANG_MAMBA_CONV_DTYPE.clear()
        envs.SGLANG_MAMBA_SSM_DTYPE.clear()

    def tearDown(self):
        envs.SGLANG_MAMBA_CONV_DTYPE.clear()
        envs.SGLANG_MAMBA_SSM_DTYPE.clear()

    def test_fp16_checkpoint_uses_fp16_convolution_state(self):
        config = SimpleNamespace(dtype="float16")
        self.assertEqual(mamba2_state_dtype(config).conv, torch.float16)

    def test_bf16_checkpoint_uses_bf16_convolution_state(self):
        config = SimpleNamespace(dtype="bfloat16")
        self.assertEqual(mamba2_state_dtype(config).conv, torch.bfloat16)

    def test_nested_text_config_supplies_dtype(self):
        config = SimpleNamespace(text_config=SimpleNamespace(torch_dtype="float16"))
        self.assertEqual(mamba2_state_dtype(config).conv, torch.float16)

    def test_torch_dtype_object_is_supported(self):
        config = SimpleNamespace(dtype=torch.float16)
        self.assertEqual(mamba2_state_dtype(config).conv, torch.float16)

    def test_explicit_environment_override_has_priority(self):
        config = SimpleNamespace(dtype="bfloat16")
        with envs.SGLANG_MAMBA_CONV_DTYPE.override("float16"):
            self.assertEqual(mamba2_state_dtype(config).conv, torch.float16)

    def test_invalid_environment_override_uses_config(self):
        config = SimpleNamespace(dtype="float16")
        with envs.SGLANG_MAMBA_CONV_DTYPE.override("invalid"):
            with self.assertLogs(
                "sglang.srt.configs.mamba_utils", level="WARNING"
            ):
                self.assertEqual(mamba2_state_dtype(config).conv, torch.float16)

    def test_missing_config_and_environment_default_to_bf16(self):
        self.assertEqual(mamba2_state_dtype().conv, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
