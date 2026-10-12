"""Three-shape sparse-MLA timings with a named rounded-probability Torch control.

Prior operator correctness is required. The control uses donor-style global
online softmax in 512-row chunks, with BF16 probabilities before FP32 PV;
the candidate uses 64-row chunks and split-K. Strict BF16 parity is reported
separately because these intermediate rounding policies can differ.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import triton
from bench_sm80_paged_mqa_logits import _group_ms, _summary
from sglang.kernels.ops.attention.dsa import sparse_mla_decode_sm80 as candidate
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=20,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
    disabled="Manual SM80 qualification; standard runner is not SM80",
)

_SHAPES = ((1, 8, 2048), (1, 16, 2048), (4, 32, 2048))
_SCALE = 0.0625


@dataclass(frozen=True)
class _Inputs:
    nope: torch.Tensor
    rope: torch.Tensor
    pool: torch.Tensor
    indices: torch.Tensor
    digest: str


def _inputs(tokens: int, heads: int, topk: int, *, device: str = "cuda") -> _Inputs:
    generator = torch.Generator().manual_seed(42)
    nope = torch.randn(tokens, heads, 512, generator=generator).bfloat16()
    rope = (torch.randn(tokens, heads, 64, generator=generator) * 4).bfloat16()
    pool = torch.randn(tokens * topk, 1, 576, generator=generator).bfloat16()
    indices = (
        torch.arange(tokens * topk, dtype=torch.int32)
        .view(tokens, 1, topk)
        .flip(-1)
        .contiguous()
    )
    digest = hashlib.sha256()
    for tensor in (nope, rope, pool, indices):
        digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return _Inputs(
        nope.to(device),
        rope.to(device),
        pool.to(device),
        indices.to(device),
        digest.hexdigest(),
    )


def _torch_rounded_probability_control(inputs: _Inputs) -> torch.Tensor:
    queries, rope_queries = inputs.nope.float(), inputs.rope.float()
    tokens, heads, _ = queries.shape
    maximum = torch.full((tokens, heads), -torch.inf, device=queries.device)
    denominator = torch.zeros_like(maximum)
    accumulated = torch.zeros_like(queries)
    for start in range(0, inputs.indices.shape[-1], 512):
        selected = inputs.indices[:, 0, start : start + 512].long()
        valid = (selected >= 0) & (selected < inputs.pool.shape[0])
        keys = inputs.pool[selected.clamp(0, inputs.pool.shape[0] - 1), 0].float()
        scores = (
            torch.bmm(queries, keys[..., :512].transpose(1, 2))
            + torch.bmm(rope_queries, keys[..., 512:].transpose(1, 2))
        ) * _SCALE
        scores = scores.masked_fill(~valid[:, None, :], -torch.inf)
        new_maximum = torch.maximum(maximum, scores.amax(-1))
        safe_maximum = torch.where(new_maximum == -torch.inf, 0.0, new_maximum)
        rescale = (maximum - safe_maximum).exp()
        probabilities = (scores - safe_maximum[..., None]).exp()
        # Preserve the explicit unnormalized probability rounding before PV.
        accumulated = accumulated * rescale[..., None] + torch.bmm(
            probabilities.bfloat16().float(), keys[..., :512]
        )
        denominator = denominator * rescale + probabilities.sum(-1)
        maximum = new_maximum
    result = accumulated / denominator.clamp_min(1e-30)[..., None]
    return torch.where(denominator[..., None] > 0, result, 0.0).bfloat16().unsqueeze(0)


def _candidate(inputs: _Inputs) -> torch.Tensor:
    return candidate.sparse_mla_decode_sm80(
        inputs.nope, inputs.rope, inputs.pool, inputs.indices, _SCALE
    )


def main() -> None:
    """Print separate parity observations and grouped operator timings as JSON."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        raise RuntimeError("This operator timing utility requires an SM80 CUDA device")
    # Ensure the named FP32 control does not silently substitute TF32 arithmetic.
    torch.backends.cuda.matmul.allow_tf32 = False
    smoke = torch.arange(256, dtype=torch.float32, device="cuda")
    first, last = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    first.record()
    smoke.add_(1)
    last.record()
    last.synchronize()
    if (
        not torch.equal(smoke.cpu(), torch.arange(256, dtype=torch.float32) + 1)
        or first.elapsed_time(last) <= 0
    ):
        raise RuntimeError("Checked CUDA work/event timing failed")
    native_root = Path(__file__).resolve().parents[5]
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    record: dict[str, object] = {
        "scope": "operator-only; weighted exp remains diagnostic; no full-model semantic or throughput claim",
        "control": "Torch FP32 online softmax, 512-row chunks, unnormalized probability BF16 cast, FP32 PV, no split-K",
        "candidate_policy": "64-row chunks, device-derived split-K, BF16 probability cast and result",
        "native_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=native_root, text=True
        ).strip(),
        "native_dirty": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=native_root, text=True
        ).strip(),
        "utility_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "timing_helper_sha256": hashlib.sha256(
            Path(_group_ms.__code__.co_filename).read_bytes()
        ).hexdigest(),
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
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "seed": 42,
        "latent_dim": 512,
        "rope_dim": 64,
        "softmax_scale": _SCALE,
        "scale_provenance": {
            "model": "Tech2wild/GLM-5.3-Int4-Int8Mix",
            "config_sha256": "238781b1049848b65be244adda218845c1a63a4aa49c4941eb59a05dec681f3b",
            "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64,
            "rope_scaling": None,
            "formula": "(192+64)^(-0.5)",
        },
        "warmups_per_arm": 10,
        "measured_calls_per_arm": 30,
        "groups": 5,
        "method": "CUDA events on current stream; six calls per group",
        "event_smoke_ms": first.elapsed_time(last),
    }
    cases: list[dict[str, object]] = []
    maximum_spread = 0.0
    for tokens, heads, topk in _SHAPES:
        inputs = _inputs(tokens, heads, topk)
        control, actual = _torch_rounded_probability_control(inputs), _candidate(inputs)
        torch.cuda.synchronize()
        if not bool(torch.isfinite(control).all()) or not bool(
            torch.isfinite(actual).all()
        ):
            raise RuntimeError(
                "Nonfinite operator result; clean timing is not admitted"
            )
        parity = {
            "strict_bf16_output_equal": torch.equal(control, actual),
            "different_elements": int((control != actual).sum()),
            "max_abs_difference": float((control.float() - actual.float()).abs().max()),
        }
        operations = {
            "control": partial(_torch_rounded_probability_control, inputs),
            "candidate": partial(_candidate, inputs),
        }
        for operation in operations.values():
            for _ in range(10):
                operation()
        torch.cuda.synchronize()
        times: dict[str, list[float]] = {name: [] for name in operations}
        for group in range(5):
            order = (
                ("control", "candidate") if group % 2 == 0 else ("candidate", "control")
            )
            for name in order:
                times[name].append(_group_ms(operations[name]))
        for samples in times.values():
            maximum_spread = max(
                maximum_spread,
                (max(samples) - min(samples)) / statistics.median(samples),
            )
        splits, length = candidate._split_k_config(
            tokens, heads, topk, properties.multi_processor_count
        )
        cases.append(
            {
                "tokens": tokens,
                "heads": heads,
                "topk": topk,
                "candidate_splits": splits,
                "candidate_split_length": length,
                "represented_input_sha256": inputs.digest,
                "parity": parity,
                "timings": {name: _summary(samples) for name, samples in times.items()},
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
