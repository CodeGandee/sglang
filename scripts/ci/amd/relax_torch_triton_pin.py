#!/usr/bin/env python3
"""Drop the exact Triton version that AMD's ROCm torch wheels require.

Those wheels pin the Triton they were built against (`triton==3.5.1` on the
ROCm 7.2.0 base, `triton==3.5.1+rocm7.2.4.gita272dfa8` on 7.2.4), and the ROCm
image replaces it with AITER's pinned Triton. Left alone, the shipped image
records a requirement that is not installed and, because the pinned
distribution exists only on repo.radeon.com, a later `pip install` in that
image either fails to resolve or "repairs" torch by pulling the CUDA build from
PyPI.

Requiring `triton` without a version keeps the requirement satisfied by
whichever Triton the image ends up with. The requirement's environment markers
are preserved.

This edits the installed dist-info rather than the wheel: bases that pip-install
torch from AMD's wheel index leave no wheel behind at all. The METADATA hash
recorded in RECORD goes stale, which pip does not verify after installation.
"""

from __future__ import annotations

import re
import sys
from importlib import metadata
from pathlib import Path

TRITON_PIN = re.compile(
    r"^(?P<name>Requires-Dist:\s*triton)\s*==\s*[^\s;]+(?P<markers>.*)$",
    re.MULTILINE,
)


def relax(metadata_text: str) -> tuple[str, list[str]]:
    """Return the text with exact triton pins dropped, plus the lines replaced."""
    replaced = [m.group(0) for m in TRITON_PIN.finditer(metadata_text)]
    return TRITON_PIN.sub(r"\g<name>\g<markers>", metadata_text), replaced


def torch_metadata_path() -> Path:
    dist = metadata.distribution("torch")
    root = Path(dist.locate_file(""))
    for pattern in (
        f"torch-{dist.version}.dist-info/METADATA",
        "torch-*.dist-info/METADATA",
    ):
        found = sorted(root.glob(pattern))
        if found:
            return found[0]
    raise SystemExit(f"FATAL: no torch dist-info found under {root}")


def main() -> int:
    try:
        version = metadata.version("torch")
    except metadata.PackageNotFoundError:
        raise SystemExit("FATAL: torch is not installed")

    path = torch_metadata_path()
    before = path.read_text(encoding="utf-8")
    after, replaced = relax(before)
    if not replaced:
        print(f"torch {version} pins no exact Triton version; nothing to relax")
        return 0

    path.write_text(after, encoding="utf-8")
    for line in replaced:
        print(f"relaxed {line!r}")

    if TRITON_PIN.search(path.read_text(encoding="utf-8")):
        raise SystemExit(f"FATAL: {path} still pins an exact Triton version")
    print(f"patched {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
