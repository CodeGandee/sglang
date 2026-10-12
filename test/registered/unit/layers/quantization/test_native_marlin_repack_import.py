"""Native Marlin repacking must not depend on optional legacy GPTQ exports."""

from __future__ import annotations

import builtins
import importlib
import importlib.util
import sys
import unittest
from collections.abc import Sequence
from types import ModuleType
from unittest.mock import patch

from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _import_owner(cuda_platform: bool) -> ModuleType:
    # Initialize the ordinary config import path first. Importing this facade
    # before its quantization dependencies exposes a separate existing cycle.
    importlib.import_module(
        "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors"
    )
    owner = importlib.import_module(
        "sglang.srt.hardware_backend.gpu.quantization.gptq_kernels"
    )
    name = "_native_marlin_import_regression"
    spec = importlib.util.spec_from_file_location(name, owner.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    original_import = builtins.__import__

    def without_legacy(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "sgl_kernel" and {"gptq_gemm", "gptq_shuffle"}.intersection(
            fromlist
        ):
            raise ImportError("Regression fixture: optional legacy GPTQ exports absent")
        return original_import(name, globals, locals, fromlist, level)

    sys.modules[name] = module
    try:
        with (
            patch("sglang.srt.utils.is_cuda", return_value=cuda_platform),
            patch("builtins.__import__", side_effect=without_legacy),
        ):
            spec.loader.exec_module(module)
    finally:
        del sys.modules[name]
    return module


class TestNativeMarlinRepackImport(CustomTestCase):
    def test_native_repack_available_when_legacy_exports_missing(self) -> None:
        module = _import_owner(cuda_platform=True)
        self.assertIs(module.gptq_marlin_repack, gptq_marlin_repack)
        self.assertIs(module.gptq_gemm, module._unsupported_kernel)
        self.assertIs(module.gptq_shuffle, module._unsupported_kernel)

    def test_non_cuda_import_retains_explicit_unavailable_behavior(self) -> None:
        module = _import_owner(cuda_platform=False)
        self.assertIs(module.gptq_marlin_repack, module._unsupported_kernel)
        with self.assertRaisesRegex(RuntimeError, "GPTQ CUDA kernels are unavailable"):
            module.gptq_marlin_repack()


if __name__ == "__main__":
    unittest.main()
