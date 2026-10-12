"""Three-shape SM80 operator timings against a standalone Torch BF16 fallback.

Correctness must be checked separately before running this utility. Parity
observations precede clean timing; no FP64, tensor dumps, or profiler are used.
Run this file directly and retain its JSON stdout with the exact invocation.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import triton
from sglang.kernels.ops.attention.dsa import triton_paged_mqa_logits_sm80 as candidate
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=15,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
    disabled="Manual SM80 qualification; standard runner is not SM80",
)

_SHAPES = ((1, 32, 2048), (1, 32, 16384), (4, 32, 16384))
_WARMUPS = 10
_GROUPS = 5
_CALLS_PER_GROUP = 6


@dataclass(frozen=True)
class _Inputs:
    q: torch.Tensor
    cache: torch.Tensor
    weights: torch.Tensor
    lengths: torch.Tensor
    table: torch.Tensor
    width: int
    digest: str


def _inputs(batch: int, heads: int, width: int) -> _Inputs:
    generator = torch.Generator().manual_seed(42)
    pages_per_request = width // 64
    pages = batch * pages_per_request
    keys = torch.randn(pages, 64, 128, generator=generator).to(torch.float8_e4m3fn)
    scales = torch.rand(pages, 64, generator=generator) + 0.5
    q = torch.randn(batch, heads, 128, generator=generator).to(torch.float8_e4m3fn)
    weights = torch.randn(batch, heads, generator=generator) / 32
    raw = torch.empty(pages, 64 * 132, dtype=torch.uint8)
    raw[:, : 64 * 128] = keys.view(torch.uint8).flatten(1)
    raw[:, 64 * 128 :] = scales.view(torch.uint8).flatten(1)
    table = (
        torch.arange(pages, dtype=torch.int32).reshape(batch, -1).flip(1).contiguous()
    )
    lengths = torch.full((batch,), width, dtype=torch.int32)
    digest = hashlib.sha256()
    for tensor in (q, raw, weights, lengths, table):
        digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return _Inputs(
        q.cuda(),
        raw.view(pages, 64, 1, 132).cuda(),
        weights.cuda(),
        lengths.cuda(),
        table.cuda(),
        width,
        digest.hexdigest(),
    )


def _torch_bf16_fallback(inputs: _Inputs) -> torch.Tensor:
    # Independent operator control reflecting the donor's BF16-matmul-output
    # rounding. Gather each 2048-token tile, then apply FP32 head gates/scales.
    # This is a named standalone control, not a working native model baseline.
    raw = inputs.cache.view(torch.uint8).flatten(1)
    key_view = raw[:, : 64 * 128].reshape(-1, 64, 128).view(torch.float8_e4m3fn)
    scales = raw[:, 64 * 128 :].contiguous().view(torch.float32)
    queries = inputs.q.to(torch.bfloat16)
    output = torch.empty(
        inputs.q.shape[0], inputs.width, device=inputs.q.device, dtype=torch.float32
    )
    for start in range(0, inputs.width, 2048):
        end = min(inputs.width, start + 2048)
        pages = inputs.table[:, start // 64 : (end + 63) // 64].long()
        keys = key_view[pages].reshape(inputs.q.shape[0], -1, 128).to(torch.bfloat16)
        dots = torch.bmm(queries, keys.transpose(1, 2)).relu().float()
        scores = torch.einsum("bhs,bh->bs", dots, inputs.weights)
        scores *= scales[pages].reshape(inputs.q.shape[0], -1)
        output[:, start:end] = scores
    return output


def _candidate(inputs: _Inputs) -> torch.Tensor:
    return candidate.triton_paged_mqa_logits(
        inputs.q,
        inputs.cache,
        inputs.weights,
        inputs.lengths,
        inputs.table,
        inputs.width,
    )


def _group_ms(operation: Callable[[], torch.Tensor]) -> float:
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(_CALLS_PER_GROUP):
        operation()
    end.record()
    end.synchronize()
    elapsed = start.elapsed_time(end)
    if elapsed <= 0:
        raise RuntimeError(
            "CUDA event timing has insufficient resolution for the selected group"
        )
    return elapsed / _CALLS_PER_GROUP


def _summary(times: list[float]) -> dict[str, float | list[float]]:
    median = statistics.median(times)
    return {
        "group_mean_ms": times,
        "median_ms": median,
        "relative_range": (max(times) - min(times)) / median,
    }


def main() -> None:
    """Print parity observations and clean grouped timings as one JSON record."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        raise RuntimeError("This operator timing utility requires an SM80 CUDA device")
    # Checked work qualifies event availability on this selected stream/device.
    smoke = torch.arange(256, device="cuda", dtype=torch.float32)
    check_start, check_end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    check_start.record()
    smoke.mul_(2)
    check_end.record()
    check_end.synchronize()
    if (
        not torch.equal(smoke.cpu(), torch.arange(256, dtype=torch.float32) * 2)
        or check_start.elapsed_time(check_end) <= 0
    ):
        raise RuntimeError("Checked CUDA work/event timing failed")
    native_root = Path(__file__).resolve().parents[5]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=native_root, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=native_root, text=True
    ).strip()
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    record: dict[str, object] = {
        "scope": "operator-only; no serving-throughput claim",
        "control": "standalone Torch BF16 matmul + FP32 head reduction, 2048-token tiles",
        "native_revision": revision,
        "native_dirty": dirty,
        "utility_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "kernel_sha256": hashlib.sha256(
            Path(candidate.__file__).read_bytes()
        ).hexdigest(),
        "kernel_import": candidate.__file__,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "gpu": properties.name,
        "gpu_properties": str(properties),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "seed": 42,
        "warmups_per_arm": _WARMUPS,
        "measured_calls_per_arm": _GROUPS * _CALLS_PER_GROUP,
        "groups": _GROUPS,
        "method": "CUDA events on current stream; six calls per group",
        "event_smoke_ms": check_start.elapsed_time(check_end),
    }
    cases: list[dict[str, object]] = []
    maximum_spread = 0.0
    for batch, heads, width in _SHAPES:
        inputs = _inputs(batch, heads, width)
        control, actual = _torch_bf16_fallback(inputs), _candidate(inputs)
        torch.cuda.synchronize()
        if not bool(torch.isfinite(control).all()) or not bool(
            torch.isfinite(actual).all()
        ):
            raise RuntimeError(
                "Nonfinite operator result; clean timing is not admitted"
            )
        control_topk = torch.topk(control, 2048, dim=1).indices
        actual_topk = torch.topk(actual, 2048, dim=1).indices
        # This separate parity record cannot waive the prior correctness gate.
        parity: dict[str, object] = {
            "strict_fp32_logits_equal": torch.equal(control, actual),
            "max_abs_logit_difference": float((control - actual).abs().max()),
            "strict_topk_indices_equal": torch.equal(control_topk, actual_topk),
            "topk_membership_differences": [
                len(
                    set(control_topk[row].cpu().tolist())
                    - set(actual_topk[row].cpu().tolist())
                )
                for row in range(batch)
            ],
        }
        operations: dict[str, Callable[[], torch.Tensor]] = {
            "control": partial(_torch_bf16_fallback, inputs),
            "candidate": partial(_candidate, inputs),
        }
        for operation in operations.values():
            for _ in range(_WARMUPS):
                operation()
        torch.cuda.synchronize()
        times: dict[str, list[float]] = {name: [] for name in operations}
        for group in range(_GROUPS):
            order = (
                ("control", "candidate") if group % 2 == 0 else ("candidate", "control")
            )
            for name in order:
                times[name].append(_group_ms(operations[name]))
        summaries = {name: _summary(samples) for name, samples in times.items()}
        for samples in times.values():
            maximum_spread = max(
                maximum_spread,
                (max(samples) - min(samples)) / statistics.median(samples),
            )
        cases.append(
            {
                "batch": batch,
                "heads": heads,
                "length": width,
                "represented_input_sha256": inputs.digest,
                "parity": parity,
                "timings": summaries,
                "speedup_over_named_control": statistics.median(times["control"])
                / statistics.median(times["candidate"]),
            }
        )
    record["environment_classification"] = (
        "stable"
        if maximum_spread <= 0.10
        else "degraded"
        if maximum_spread <= 0.50
        else "unstable"
    )
    record["cases"] = cases
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
