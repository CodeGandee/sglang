"""Bounded ordinary-prefill scoring for the SM80 DSA indexer.

Derived from the ragged Torch formula in predkv-inference donor revision
e5e38b9813908e69b35d1660b2d116337a714cc9, scripts/glm53-a100-sglang/sitecustomize.py.
The BF16 dot-result cast is part of this operator's declared numerical policy.
"""

from __future__ import annotations

import torch

_DIM = 128
_TEMP_BYTES = 32 * 1024 * 1024
_KEY_TILE = 2048
_SCORE_BYTES = 32


def _require_sm80(device: torch.device) -> None:
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (8, 0):
        raise ValueError("SM80 ragged MQA logits require CUDA compute capability 8.0.")


def _tile_shape(rows: int, heads: int, keys: int) -> tuple[int, int]:
    # FP64 verification can transiently hold a dot, its BF16 cast, and an FP64
    # recast, weighted products and reduced rows; reserve 32 bytes/score plus
    # converted q/k operands in both modes.
    # This bounds score/conversion temporaries, excluding output and BLAS workspace.
    key_tile = max(
        1, min(keys, _KEY_TILE, _TEMP_BYTES // max(1, heads * _SCORE_BYTES + _DIM * 8))
    )
    remaining = _TEMP_BYTES - key_tile * _DIM * 8
    row_bytes = heads * (_SCORE_BYTES * key_tile + _DIM * 8 + 8)
    if row_bytes > remaining:
        raise ValueError("Head count exceeds the SM80 ragged-score temporary budget.")
    return max(1, min(rows, remaining // max(1, row_bytes))), key_tile


def torch_ragged_mqa_logits(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scales: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    *,
    _verification_fp64: bool = False,
    _verification_precast: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute represented DSA scores for ordinary resident SM80 prefill.

    Parameters
    ----------
    q_fp8 : torch.Tensor
        E4M3FN queries ``[T,H,128]`` with unit dimension stride and H > 0.
    k_fp8 : torch.Tensor
        Gathered E4M3FN keys ``[S,128]`` with unit dimension stride.
    k_scales : torch.Tensor
        FP32 key scales ``[S]``.
    weights : torch.Tensor
        FP32 head gates ``[T,H]``, already including query scales.
    ks, ke : torch.Tensor
        INT32/INT64 half-open valid key ranges ``[T]``. Ranges intersect the
        represented key extent; reversed and empty ranges produce masked rows.
    _verification_fp64 : bool, optional
        Private eager arithmetic widening. Preserves the BF16 dot-result cast
        before ReLU and the FP32 result boundary.
    _verification_precast : torch.Tensor, optional
        Private contiguous FP64 ``[T,S]`` sink with the query device. Requires
        FP64 verification; records actual pre-cast scores alongside FP32 output.

    Returns
    -------
    torch.Tensor
        FP32 ``[T,S]`` logits, with negative infinity outside each valid range.
    """
    _require_sm80(q_fp8.device)
    if q_fp8.ndim != 3 or q_fp8.shape[-1] != _DIM or q_fp8.shape[1] == 0:
        raise ValueError("SM80 ragged logits require queries shaped [T,H,128], H > 0.")
    if k_fp8.ndim != 2 or k_fp8.shape[-1] != _DIM:
        raise ValueError("SM80 ragged logits require keys shaped [S,128].")
    rows, heads, _ = q_fp8.shape
    keys = k_fp8.shape[0]
    if q_fp8.dtype != torch.float8_e4m3fn or k_fp8.dtype != torch.float8_e4m3fn:
        raise ValueError("SM80 ragged logits require E4M3FN query/key inputs.")
    if weights.shape != (rows, heads) or weights.dtype != torch.float32:
        raise ValueError("SM80 ragged logits require FP32 head gates shaped [T,H].")
    if k_scales.shape != (keys,) or k_scales.dtype != torch.float32:
        raise ValueError("SM80 ragged logits require FP32 key scales shaped [S].")
    if any(
        tensor.shape != (rows,) or tensor.dtype not in (torch.int32, torch.int64)
        for tensor in (ks, ke)
    ):
        raise ValueError("SM80 ragged logits require integer ranges shaped [T].")
    if any(
        tensor.device != q_fp8.device for tensor in (k_fp8, k_scales, weights, ks, ke)
    ):
        raise ValueError("SM80 ragged logits require all inputs on the query device.")
    if q_fp8.stride(-1) != 1 or k_fp8.stride(-1) != 1:
        raise ValueError("SM80 ragged logits require unit query/key dimension strides.")
    if _verification_fp64 and torch.cuda.is_current_stream_capturing():
        raise ValueError("SM80 ragged logits FP64 verification is eager-only.")
    if _verification_precast is not None and (
        not _verification_fp64
        or _verification_precast.shape != (rows, keys)
        or _verification_precast.dtype != torch.float64
        or _verification_precast.device != q_fp8.device
        or not _verification_precast.is_contiguous()
    ):
        raise ValueError(
            "Pre-cast capture requires FP64 verification and a contiguous FP64 [T,S] sink on the query device."
        )

    output = torch.full(
        (rows, keys), -torch.inf, dtype=torch.float32, device=q_fp8.device
    )
    if rows == 0 or keys == 0:
        return output
    row_tile, key_tile = _tile_shape(rows, heads, keys)
    compute_dtype = torch.float64 if _verification_fp64 else torch.float32
    dot_dtype = torch.float64 if _verification_fp64 else torch.bfloat16
    for row_start in range(0, rows, row_tile):
        row_end = min(rows, row_start + row_tile)
        q = q_fp8[row_start:row_end].to(dot_dtype)
        gates = weights[row_start:row_end].to(compute_dtype)
        for key_start in range(0, keys, key_tile):
            key_end = min(keys, key_start + key_tile)
            k = k_fp8[key_start:key_end].to(dot_dtype)
            # Keep this cast even in the widened actual-candidate path.
            dot = torch.matmul(q, k.t()).to(torch.bfloat16).relu_()
            # Explicit reduction prevents an einsum-to-FP32-BMM lowering from
            # introducing TF32 when the surrounding runtime enables it.
            block = (dot.to(compute_dtype) * gates[..., None]).sum(dim=1)
            block = block * k_scales[key_start:key_end].to(compute_dtype)
            positions = torch.arange(key_start, key_end, device=q_fp8.device)
            valid = (positions[None, :] >= ks[row_start:row_end, None]) & (
                positions[None, :] < ke[row_start:row_end, None]
            )
            block.masked_fill_(~valid, -torch.inf)
            if _verification_precast is not None:
                _verification_precast[row_start:row_end, key_start:key_end] = block
            output[row_start:row_end, key_start:key_end] = block.to(torch.float32)
            del dot, block, k
    return output


__all__ = ["torch_ragged_mqa_logits"]
