import subprocess
import tempfile
from pathlib import Path

import pytest

AOT_ROOT = Path(__file__).resolve().parents[1]
SHADOWKV_CMAKE = AOT_ROOT / "cmake/shadowkv.cmake"


def _configure(*, enabled: bool, cuda_arch: str) -> subprocess.CompletedProcess[str]:
    assignments = "\n".join(
        (
            f"set(SGL_KERNEL_ENABLE_SHADOWKV {'ON' if enabled else 'OFF'})",
            f"set(SGL_KERNEL_CUDA_ARCH {cuda_arch})",
            "set(SGL_KERNEL_BUILD_SM90_VARIANT OFF)",
            "set(SGL_KERNEL_BUILD_SM100_VARIANT ON)",
            "set(SGL_KERNEL_ENABLE_BF16 ON)",
        )
    )
    source = (
        f'set(PROJECT_SOURCE_DIR "{AOT_ROOT.as_posix()}")\n'
        f'include("{SHADOWKV_CMAKE.as_posix()}")\n'
        f"{assignments}\n"
        "sgl_configure_shadowkv_sources(CUDA_SOURCES BINDING_SOURCES SPECIALIZED_SOURCES)\n"
        'message(STATUS "cuda=${CUDA_SOURCES}")\n'
        'message(STATUS "bindings=${BINDING_SOURCES}")\n'
        'message(STATUS "specialized=${SPECIALIZED_SOURCES}")\n'
    )
    with tempfile.TemporaryDirectory() as directory:
        script = Path(directory) / "configure.cmake"
        script.write_text(source, encoding="utf-8")
        return subprocess.run(
            ("cmake", "-P", str(script)),
            check=False,
            capture_output=True,
            text=True,
        )


@pytest.mark.parametrize("cuda_arch", ["80", "100a"])
def test_supported_profiles_export_only_core_shadowkv_sources(cuda_arch):
    result = _configure(enabled=True, cuda_arch=cuda_arch)
    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    for source in ("packed_gqa.cu", "plan_reuse.cu", "reconstruct_rope.cu"):
        assert f"generic/{source}" in output
    assert "bindings/shadowkv_extension.cc" in output
    assert "specialized=" in output


def test_disabled_profile_exports_no_shadowkv_sources():
    result = _configure(enabled=False, cuda_arch="90a")
    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert "packed_gqa.cu" not in output
    assert "shadowkv_extension.cc" not in output


def test_unsupported_profile_fails_configuration():
    result = _configure(enabled=True, cuda_arch="90a")
    assert result.returncode != 0
    assert "support only" in result.stderr


def test_source_tree_contains_no_retired_specializations():
    source_root = AOT_ROOT / "csrc/shadowkv"
    assert {
        path.relative_to(source_root).as_posix() for path in source_root.rglob("*.cu")
    } == {
        "generic/packed_gqa.cu",
        "generic/plan_reuse.cu",
        "generic/reconstruct_rope.cu",
    }
    binding = source_root / "bindings/shadowkv_extension.cc"
    assert binding.is_file()
    text = binding.read_text(encoding="utf-8")
    assert text.count("m.def(") == 4
    assert "plan_device" not in text


def test_optional_extension_is_separate_from_common_ops():
    cmake = (AOT_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    common_extension = (AOT_ROOT / "csrc/common_extension.cc").read_text(
        encoding="utf-8"
    )
    assert "Python_add_library(\n        shadowkv_ops" in cmake
    assert "shadowkv_reconstruct" not in common_extension


def test_public_module_exports_only_four_shadowkv_operations():
    package_init = (AOT_ROOT / "python/sgl_kernel/__init__.py").read_text(
        encoding="utf-8"
    )
    assert "shadowkv_reconstruct," in package_init
    assert "shadowkv_reconstruct_rope," in package_init
    assert "shadowkv_plan_reuse," in package_init
    assert "shadowkv_packed_gqa," in package_init
    assert "shadowkv_plan_device" not in package_init
