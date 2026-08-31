import subprocess
import tempfile
from pathlib import Path

import pytest

AOT_ROOT = Path(__file__).resolve().parents[1]
SHADOWKV_CMAKE = AOT_ROOT / "cmake/shadowkv.cmake"


def _configure(
    *,
    enabled: bool,
    cuda_arch: str,
    sm90_variant: bool,
    sm100_variant: bool,
    bf16: bool = True,
    specialization_arch: str = "",
    specialization_sources: str = "",
) -> subprocess.CompletedProcess[str]:
    values = {
        "SGL_KERNEL_ENABLE_SHADOWKV": enabled,
        "SGL_KERNEL_CUDA_ARCH": cuda_arch,
        "SGL_KERNEL_BUILD_SM90_VARIANT": sm90_variant,
        "SGL_KERNEL_BUILD_SM100_VARIANT": sm100_variant,
        "SGL_KERNEL_ENABLE_BF16": bf16,
        "SGL_KERNEL_SHADOWKV_SPECIALIZATION_ARCH": specialization_arch,
        "SGL_KERNEL_SHADOWKV_SPECIALIZATION_SOURCES": specialization_sources,
    }
    assignments = "\n".join(
        f"set({name} {'ON' if value is True else 'OFF' if value is False else value})"
        for name, value in values.items()
    )
    source = (
        assignments
        + f'\ninclude("{SHADOWKV_CMAKE.as_posix()}")\n'
        + "sgl_configure_shadowkv_sources(SHADOWKV_GENERIC_SOURCES SHADOWKV_BINDING_SOURCES SHADOWKV_SPECIALIZED_SOURCES)\n"
        + 'message(STATUS "shadowkv-generic=${SHADOWKV_GENERIC_SOURCES}")\n'
        + 'message(STATUS "shadowkv-bindings=${SHADOWKV_BINDING_SOURCES}")\n'
        + 'message(STATUS "shadowkv-specialized=${SHADOWKV_SPECIALIZED_SOURCES}")\n'
    )
    with tempfile.TemporaryDirectory() as directory:
        script = Path(directory) / "configure-shadowkv.cmake"
        script.write_text(source, encoding="utf-8")
        return subprocess.run(
            ("cmake", "-P", str(script)),
            check=False,
            capture_output=True,
            text=True,
        )


@pytest.mark.parametrize("cuda_arch", ["80", "100a"])
def test_enabled_precise_profiles_export_shadowkv_sources(cuda_arch):
    result = _configure(
        enabled=True,
        cuda_arch=cuda_arch,
        sm90_variant=False,
        sm100_variant=True,
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    for source in ("packed_gqa.cu", "plan_reuse.cu", "reconstruct_rope.cu"):
        assert source in output
        assert f"generic/{source}" in output
    assert "bindings/shadowkv_extension.cc" in output
    assert "shadowkv-specialized=" in output


def test_disabled_profile_exports_no_shadowkv_sources():
    result = _configure(
        enabled=False,
        cuda_arch="90a",
        sm90_variant=True,
        sm100_variant=False,
        bf16=False,
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert "shadowkv-generic=" in output
    assert "packed_gqa.cu" not in output
    assert "shadowkv_extension.cc" not in output


@pytest.mark.parametrize(
    "cuda_arch,sm90_variant,sm100_variant,bf16,message",
    [
        ("90a", False, True, True, "generic candidate"),
        ("80", True, True, True, "only the precise SM100"),
        ("100a", False, False, True, "only the precise SM100"),
        ("80", False, True, False, "requires BF16"),
    ],
)
def test_enabled_inconsistent_profiles_fail_configuration(
    cuda_arch, sm90_variant, sm100_variant, bf16, message
):
    result = _configure(
        enabled=True,
        cuda_arch=cuda_arch,
        sm90_variant=sm90_variant,
        sm100_variant=sm100_variant,
        bf16=bf16,
    )
    assert result.returncode != 0
    assert message in " ".join(result.stderr.split())


@pytest.mark.parametrize(
    "specialization_arch,specialization_sources,message",
    [
        ("80", "", "requires a non-empty specialized source set"),
        ("", "csrc/shadowkv/sm80/example.cu", "require an exact specialization"),
        (
            "100a",
            "csrc/shadowkv/sm100a/example.cu",
            "contradicts build target",
        ),
    ],
)
def test_contradictory_specialization_configuration_fails(
    specialization_arch, specialization_sources, message
):
    result = _configure(
        enabled=True,
        cuda_arch="80",
        sm90_variant=False,
        sm100_variant=True,
        specialization_arch=specialization_arch,
        specialization_sources=specialization_sources,
    )
    assert result.returncode != 0
    assert message in " ".join(result.stderr.split())


def test_generic_and_specialized_source_layout_is_explicit():
    generic = AOT_ROOT / "csrc/shadowkv/generic"
    common = AOT_ROOT / "csrc/shadowkv/common"
    bindings = AOT_ROOT / "csrc/shadowkv/bindings"
    assert {path.name for path in generic.glob("*.cu")} == {
        "packed_gqa.cu",
        "plan_reuse.cu",
        "reconstruct_rope.cu",
    }
    assert (common / "device_contract.cuh").is_file()
    assert (common / "operations.h").is_file()
    assert (bindings / "shadowkv_extension.cc").is_file()
    for architecture in ("sm80", "sm90a", "sm100a"):
        directory = AOT_ROOT / f"csrc/shadowkv/{architecture}"
        assert directory.is_dir()
        assert list(directory.glob("*.cu")) == []
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in generic.glob("*.cu")
    )
    assert "__CUDA_ARCH__" not in source


def test_packed_gqa_shared_reductions_use_explicit_warp_barriers():
    source = (AOT_ROOT / "csrc/shadowkv/generic/packed_gqa.cu").read_text(
        encoding="utf-8"
    )
    assert source.count("__syncwarp();") == 3
    assert "before lane zero reuses reduction[0]" in source
    assert "overwrites reduction[0] with the block total" in source


def test_shadowkv_extension_is_separate_from_common_ops():
    cmake = (AOT_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    common_extension = (AOT_ROOT / "csrc/common_extension.cc").read_text(
        encoding="utf-8"
    )
    binding = (AOT_ROOT / "csrc/shadowkv/bindings/shadowkv_extension.cc").read_text(
        encoding="utf-8"
    )
    assert "Python_add_library(\n        shadowkv_ops" in cmake
    assert (
        "${SHADOWKV_GENERIC_SOURCES}"
        not in cmake.split("# ======================= Optional ShadowKV Build")[0]
    )
    assert "shadowkv_reconstruct" not in common_extension
    assert "shadowkv_reconstruct_generic_aot_v1" in binding
    assert "REGISTER_EXTENSION(shadowkv_ops)" in binding
