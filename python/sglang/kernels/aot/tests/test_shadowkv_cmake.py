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
    target_architecture: str | None = None,
    build_input_sha256: str | None = None,
    bundle_ids: str | None = None,
    native_sources: str | None = None,
    expected_symbols: str | None = None,
) -> subprocess.CompletedProcess[str]:
    if enabled:
        if target_architecture is None:
            target_architecture = f"sm{cuda_arch}"
        if build_input_sha256 is None:
            build_input_sha256 = "a" * 64
        if bundle_ids is None:
            bundle_ids = "shadowkv.runtime.generic-aot.v1"
        if native_sources is None:
            native_sources = ";".join(
                (
                    "csrc/shadowkv/bindings/shadowkv_extension.cc",
                    "csrc/shadowkv/generic/packed_gqa.cu",
                    "csrc/shadowkv/generic/plan_reuse.cu",
                    "csrc/shadowkv/generic/reconstruct_rope.cu",
                )
            )
        if expected_symbols is None:
            expected_symbols = "shadowkv_packed_gqa_generic_aot_v1"
    values = {
        "SGL_KERNEL_ENABLE_SHADOWKV": enabled,
        "SGL_KERNEL_CUDA_ARCH": cuda_arch,
        "SGL_KERNEL_BUILD_SM90_VARIANT": sm90_variant,
        "SGL_KERNEL_BUILD_SM100_VARIANT": sm100_variant,
        "SGL_KERNEL_ENABLE_BF16": bf16,
        "SGL_KERNEL_SHADOWKV_TARGET_ARCHITECTURE": target_architecture or "",
        "SGL_KERNEL_SHADOWKV_BUILD_INPUT_SHA256": build_input_sha256 or "",
        "SGL_KERNEL_SHADOWKV_BUNDLE_IDS": bundle_ids or "",
        "SGL_KERNEL_SHADOWKV_NATIVE_SOURCES": native_sources or "",
        "SGL_KERNEL_SHADOWKV_EXPECTED_SYMBOLS": expected_symbols or "",
    }
    assignments = "\n".join(
        f"set({name} {'ON' if value is True else 'OFF' if value is False else value})"
        for name, value in values.items()
    )
    source = (
        f'set(PROJECT_SOURCE_DIR "{AOT_ROOT.as_posix()}")\n'
        + f'include("{SHADOWKV_CMAKE.as_posix()}")\n'
        + assignments
        + "\n"
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


@pytest.mark.parametrize(
    "cuda_arch,sm90_variant,sm100_variant,bf16",
    [
        ("80", False, True, True),
        ("90a", True, False, False),
        ("100a", False, True, True),
    ],
)
def test_disabled_profile_exports_no_shadowkv_sources(
    cuda_arch, sm90_variant, sm100_variant, bf16
):
    result = _configure(
        enabled=False,
        cuda_arch=cuda_arch,
        sm90_variant=sm90_variant,
        sm100_variant=sm100_variant,
        bf16=bf16,
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert "shadowkv-generic=" in output
    assert "packed_gqa.cu" not in output
    assert "shadowkv_extension.cc" not in output


@pytest.mark.parametrize(
    "cuda_arch,sm90_variant,sm100_variant,bf16,message",
    [
        ("90a", False, True, True, "contradicts"),
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
        target_architecture="sm80" if cuda_arch == "90a" else None,
    )
    assert result.returncode != 0
    assert message in " ".join(result.stderr.split())


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("build_input_sha256", "bad", "canonical build-input SHA-256"),
        ("bundle_ids", "", "at least one effective bundle"),
        ("native_sources", "../outside.cu", "Unsafe ShadowKV native source"),
        ("native_sources", "csrc/common_extension.cc", "outside csrc/shadowkv"),
        ("expected_symbols", "", "manifest-derived internal symbols"),
    ],
)
def test_invalid_manifest_derived_configuration_fails(field, value, message):
    overrides = {field: value}
    result = _configure(
        enabled=True,
        cuda_arch="80",
        sm90_variant=False,
        sm100_variant=True,
        **overrides,
    )
    assert result.returncode != 0
    assert message in " ".join(result.stderr.split())


def test_disabled_profile_rejects_stale_bundle_inputs():
    result = _configure(
        enabled=False,
        cuda_arch="80",
        sm90_variant=False,
        sm100_variant=True,
        bundle_ids="shadowkv.runtime.generic-aot.v1",
    )
    assert result.returncode != 0
    assert "stale bundle input SGL_KERNEL_SHADOWKV_BUNDLE_IDS" in " ".join(
        result.stderr.split()
    )


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
    expected_specialized_sources = {
        "sm80": {"fused_key.cu"},
        "sm90a": set(),
        "sm100a": set(),
    }
    for architecture, expected in expected_specialized_sources.items():
        directory = AOT_ROOT / f"csrc/shadowkv/{architecture}"
        assert directory.is_dir()
        assert {path.name for path in directory.glob("*.cu")} == expected
    assert (bindings / "shadowkv_sm80_fused_key_extension.cc").is_file()
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in generic.glob("*.cu")
    )
    assert "__CUDA_ARCH__" not in source


def test_packed_gqa_shared_reductions_use_explicit_warp_barriers():
    source = (AOT_ROOT / "csrc/shadowkv/generic/packed_gqa.cu").read_text(
        encoding="utf-8"
    )
    assert source.count("__syncwarp();") == 3
    assert source.count("const float result = shared[0];") == 2
    assert "until every thread has\n  // copied the published maximum" in source
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


def test_shadowkv_public_wrappers_are_exported_only_with_the_optional_extension():
    package_init = (
        AOT_ROOT / "python/sgl_kernel/__init__.py"
    ).read_text(encoding="utf-8")
    availability, conditional_exports = package_init.split(
        "if shadowkv_ops is not None:", 1
    )
    assert "def shadowkv_kernels_available():" in availability
    assert "from sgl_kernel.shadowkv import (" in conditional_exports
    assert "shadowkv_packed_gqa," in conditional_exports
