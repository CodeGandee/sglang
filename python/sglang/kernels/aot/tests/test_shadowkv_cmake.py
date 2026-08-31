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
) -> subprocess.CompletedProcess[str]:
    values = {
        "SGL_KERNEL_ENABLE_SHADOWKV": enabled,
        "SGL_KERNEL_CUDA_ARCH": cuda_arch,
        "SGL_KERNEL_BUILD_SM90_VARIANT": sm90_variant,
        "SGL_KERNEL_BUILD_SM100_VARIANT": sm100_variant,
        "SGL_KERNEL_ENABLE_BF16": bf16,
    }
    assignments = "\n".join(
        f"set({name} {'ON' if value is True else 'OFF' if value is False else value})"
        for name, value in values.items()
    )
    source = (
        assignments
        + f'\ninclude("{SHADOWKV_CMAKE.as_posix()}")\n'
        + "sgl_configure_shadowkv_sources(SHADOWKV_SOURCES)\n"
        + 'message(STATUS "shadowkv-sources=${SHADOWKV_SOURCES}")\n'
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
    assert "shadowkv-sources=" in output
    assert "packed_gqa.cu" not in output


@pytest.mark.parametrize(
    "cuda_arch,sm90_variant,sm100_variant,bf16,message",
    [
        ("90a", False, True, True, "CUDA_ARCH=80 or 100a"),
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
    assert message in result.stderr
