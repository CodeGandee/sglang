"""Explicit SM80 BF16 resident sparse-MLA decode with split-K softmax.

Derived from predkv-inference e5e38b9813908e69b35d1660b2d116337a714cc9,
scripts/explore/sm80-dsa/sparse_mla_decode_sm80.py. The native port adds input
guards and private eager FP64 arithmetic within the actual gather/rounding path.
Probability values are rounded to BF16 before the value dot; this algorithmic
boundary is preserved in the FP64 verification specialization.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

_NOPE_DIM = 512
_ROPE_DIM = 64
_KV_DIM = 576
_HEAD_TILE = 16
_KV_CHUNK = 64


@triton.jit
def _attention_exp(value, VERIFY_FP64: tl.constexpr):
    if VERIFY_FP64:
        return libdevice.exp(value)
    return tl.exp(value)


@triton.jit
def _sparse_mla_decode_splitk_kernel(
    q_nope_ptr,
    q_rope_ptr,
    kv_ptr,
    indices_ptr,
    partial_acc_ptr,
    partial_ml_ptr,
    sm_scale,
    num_kv_rows,
    q_nope_stride_t,
    q_nope_stride_h,
    q_rope_stride_t,
    q_rope_stride_h,
    kv_stride_row,
    indices_stride_t,
    acc_stride_t,
    acc_stride_s,
    acc_stride_h,
    ml_stride_t,
    ml_stride_s,
    NUM_HEADS: tl.constexpr,
    TOPK: tl.constexpr,
    SPLIT_LEN: tl.constexpr,
    HEAD_TILE: tl.constexpr,
    KV_CHUNK: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    VERIFY_FP64: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_tile = tl.program_id(1)
    split_idx = tl.program_id(2)

    head_offsets = head_tile * HEAD_TILE + tl.arange(0, HEAD_TILE)
    head_mask = head_offsets < NUM_HEADS

    nope_offsets = tl.arange(0, NOPE_DIM)
    rope_offsets = tl.arange(0, ROPE_DIM)
    q_nope = tl.load(
        q_nope_ptr
        + token_idx * q_nope_stride_t
        + head_offsets[:, None] * q_nope_stride_h
        + nope_offsets[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )
    q_rope = tl.load(
        q_rope_ptr
        + token_idx * q_rope_stride_t
        + head_offsets[:, None] * q_rope_stride_h
        + rope_offsets[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )

    split_start = split_idx * SPLIT_LEN
    split_end = tl.minimum(split_start + SPLIT_LEN, TOPK)

    compute_dtype: tl.constexpr = tl.float64 if VERIFY_FP64 else tl.float32
    scale = sm_scale.to(tl.float64) if VERIFY_FP64 else sm_scale
    running_max = tl.full((HEAD_TILE,), float("-inf"), dtype=compute_dtype)
    running_sum = tl.zeros((HEAD_TILE,), dtype=compute_dtype)
    acc = tl.zeros((HEAD_TILE, NOPE_DIM), dtype=compute_dtype)

    for chunk_start in range(split_start, split_end, KV_CHUNK):
        chunk_offsets = chunk_start + tl.arange(0, KV_CHUNK)
        in_range = chunk_offsets < split_end
        selected = tl.load(
            indices_ptr + token_idx * indices_stride_t + chunk_offsets,
            mask=in_range,
            other=-1,
        )
        valid = in_range & (selected >= 0) & (selected < num_kv_rows)
        safe = tl.where(valid, selected, 0).to(tl.int64)

        row_ptr = kv_ptr + safe[:, None] * kv_stride_row
        k_nope = tl.load(
            row_ptr + tl.arange(0, NOPE_DIM)[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        k_rope = tl.load(
            row_ptr + NOPE_DIM + tl.arange(0, ROPE_DIM)[None, :],
            mask=valid[:, None],
            other=0.0,
        )

        if VERIFY_FP64:
            scores = tl.zeros((HEAD_TILE, KV_CHUNK), dtype=tl.float64)
            for local_head in range(HEAD_TILE):
                selected_head = tl.arange(0, HEAD_TILE) == local_head
                q_main = tl.sum(
                    tl.where(selected_head[:, None], q_nope.to(tl.float64), 0.0), axis=0
                )
                q_tail = tl.sum(
                    tl.where(selected_head[:, None], q_rope.to(tl.float64), 0.0), axis=0
                )
                dot = tl.sum(k_nope.to(tl.float64) * q_main[None, :], axis=1)
                dot += tl.sum(k_rope.to(tl.float64) * q_tail[None, :], axis=1)
                scores += tl.where(selected_head[:, None], dot[None, :], 0.0)
        else:
            scores = tl.dot(q_nope, tl.trans(k_nope), out_dtype=tl.float32)
            scores += tl.dot(q_rope, tl.trans(k_rope), out_dtype=tl.float32)
        scores = scores * scale
        scores = tl.where(valid[None, :], scores, float("-inf"))

        # Online softmax update with the reference's safe-max construction:
        # exp(-inf - 0) = 0 keeps all-invalid prefixes at zero contribution.
        chunk_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, chunk_max)
        safe_max = tl.where(new_max == float("-inf"), 0.0, new_max)
        old_scale = _attention_exp(running_max - safe_max, VERIFY_FP64)
        probabilities = _attention_exp(scores - safe_max[:, None], VERIFY_FP64)

        acc = acc * old_scale[:, None]
        rounded_probabilities = probabilities.to(tl.bfloat16)
        if VERIFY_FP64:
            for local_head in range(HEAD_TILE):
                selected_head = tl.arange(0, HEAD_TILE) == local_head
                p = tl.sum(
                    tl.where(
                        selected_head[:, None],
                        rounded_probabilities.to(tl.float64),
                        0.0,
                    ),
                    axis=0,
                )
                weighted_values = tl.sum(k_nope.to(tl.float64) * p[:, None], axis=0)
                acc += tl.where(selected_head[:, None], weighted_values[None, :], 0.0)
        else:
            acc += tl.dot(rounded_probabilities, k_nope, out_dtype=tl.float32)
        running_sum = running_sum * old_scale + tl.sum(probabilities, axis=1)
        running_max = new_max

    acc_base = partial_acc_ptr + token_idx * acc_stride_t + split_idx * acc_stride_s
    tl.store(
        acc_base + head_offsets[:, None] * acc_stride_h + nope_offsets[None, :],
        acc,
        mask=head_mask[:, None],
    )
    ml_base = partial_ml_ptr + token_idx * ml_stride_t + split_idx * ml_stride_s
    tl.store(ml_base + head_offsets * 2, running_max, mask=head_mask)
    tl.store(ml_base + head_offsets * 2 + 1, running_sum, mask=head_mask)


@triton.jit
def _sparse_mla_decode_combine_kernel(
    partial_acc_ptr,
    partial_ml_ptr,
    out_ptr,
    precast_ptr,
    acc_stride_t,
    acc_stride_s,
    acc_stride_h,
    ml_stride_t,
    ml_stride_s,
    out_stride_t,
    out_stride_h,
    NUM_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    HEAD_TILE: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    VERIFY_FP64: tl.constexpr,
    CAPTURE_PRECAST: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_tile = tl.program_id(1)

    head_offsets = head_tile * HEAD_TILE + tl.arange(0, HEAD_TILE)
    head_mask = head_offsets < NUM_HEADS
    nope_offsets = tl.arange(0, NOPE_DIM)

    ml_base = partial_ml_ptr + token_idx * ml_stride_t
    split_offsets = tl.arange(0, NUM_SPLITS)
    ml_offsets = split_offsets[None, :] * ml_stride_s + head_offsets[:, None] * 2
    m_parts = tl.load(
        ml_base + ml_offsets, mask=head_mask[:, None], other=float("-inf")
    )
    l_parts = tl.load(ml_base + ml_offsets + 1, mask=head_mask[:, None], other=0.0)

    global_max = tl.max(m_parts, axis=1)
    safe_max = tl.where(global_max == float("-inf"), 0.0, global_max)
    rescale = _attention_exp(m_parts - safe_max[:, None], VERIFY_FP64)
    rescale = tl.where(m_parts == float("-inf"), 0.0, rescale)
    global_sum = tl.sum(l_parts * rescale, axis=1)

    compute_dtype: tl.constexpr = tl.float64 if VERIFY_FP64 else tl.float32
    acc = tl.zeros((HEAD_TILE, NOPE_DIM), dtype=compute_dtype)
    for split in tl.static_range(0, NUM_SPLITS):
        m_vals = tl.load(
            ml_base + split * ml_stride_s + head_offsets * 2,
            mask=head_mask,
            other=float("-inf"),
        )
        scale = _attention_exp(m_vals - safe_max, VERIFY_FP64)
        scale = tl.where(m_vals == float("-inf"), 0.0, scale)
        part = tl.load(
            partial_acc_ptr
            + token_idx * acc_stride_t
            + split * acc_stride_s
            + head_offsets[:, None] * acc_stride_h
            + nope_offsets[None, :],
            mask=head_mask[:, None],
            other=0.0,
        )
        acc += part * scale[:, None]

    normalized = acc / tl.maximum(global_sum, 1e-30)[:, None]
    normalized = tl.where((global_sum > 0)[:, None], normalized, 0.0)
    if CAPTURE_PRECAST:
        tl.store(
            precast_ptr
            + token_idx * NUM_HEADS * NOPE_DIM
            + head_offsets[:, None] * NOPE_DIM
            + nope_offsets[None, :],
            normalized,
            mask=head_mask[:, None],
        )
    tl.store(
        out_ptr
        + token_idx * out_stride_t
        + head_offsets[:, None] * out_stride_h
        + nope_offsets[None, :],
        normalized.to(tl.bfloat16),
        mask=head_mask[:, None],
    )


def _require_sm80(device: torch.device) -> None:
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (8, 0):
        raise ValueError("SM80 sparse-MLA requires CUDA compute capability 8.0.")


def _split_k_config(
    tokens: int, heads: int, topk: int, sm_count: int
) -> tuple[int, int]:
    if min(tokens, heads, topk, sm_count) <= 0:
        raise ValueError(
            "Split-K planning requires positive tokens, heads, topk and SM count."
        )
    head_tiles = triton.cdiv(heads, _HEAD_TILE)
    target = max(
        1, min(triton.cdiv(sm_count, tokens * head_tiles), triton.cdiv(topk, _KV_CHUNK))
    )
    splits = 1 << (target - 1).bit_length()
    split_len = triton.cdiv(topk, splits * _KV_CHUNK) * _KV_CHUNK
    # Retain padded power-of-two splits. Empty trailing splits publish -inf/0
    # neutral state, preserving the combine reduction for non-power-of-two tails.
    return splits, split_len


def validate_sm80_sparse_mla_config(
    *,
    device: torch.device,
    prefill_backend: str | None,
    kv_dtype: torch.dtype,
    hisparse: bool,
    speculative: bool,
) -> None:
    """Validate the initial resident SM80 decode configuration before allocation.

    Parameters
    ----------
    device : torch.device
        Execution device, which must have compute capability 8.0.
    prefill_backend : str, optional
        Independently selected prefill backend; SM80 decode is not a prefill route.
    kv_dtype : torch.dtype
        Resident cache representation, required to be BF16.
    hisparse, speculative : bool
        Whether unsupported offload or speculative execution is enabled.
    """
    _require_sm80(device)
    if prefill_backend == "triton-sm80":
        raise ValueError("SM80 sparse-MLA is a decode-only backend.")
    if kv_dtype != torch.bfloat16:
        raise ValueError("SM80 sparse-MLA requires resident BF16 KV.")
    if hisparse or speculative:
        raise ValueError(
            "SM80 sparse-MLA currently requires ordinary resident decode without HiSparse or speculation."
        )


def sparse_mla_decode_sm80(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    *,
    _verification_fp64: bool = False,
    _verification_precast: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute SM80 resident BF16 sparse-MLA with split-K attention.

    Parameters
    ----------
    q_nope, q_rope : torch.Tensor
        BF16 queries shaped ``[T, H, 512]`` / ``[T, H, 64]`` with unit
        dimension strides; sliced query rows and heads are supported.
    kv : torch.Tensor
        BF16 pool shaped ``[N, 1, 576]`` with unit dimension stride.
    indices : torch.Tensor
        int32/int64 selected pool rows shaped ``[T, 1, K]``; ``-1`` and
        out-of-range entries are ignored.
    sm_scale : float
        Finite positive softmax scale, represented as FP32 at kernel admission.
    d_v : int
        Must be 512 (the MLA absorbed value width).
    _verification_fp64 : bool, optional
        Private eager-only widening of dots, softmax, partial states and
        combination to FP64, preserving the BF16 probability cast.
    _verification_precast : torch.Tensor, optional
        Contiguous FP64 sink with the output shape and device. Requires FP64
        verification; captures actual pre-cast results alongside BF16 output.

    Returns
    -------
    torch.Tensor
        BF16 output shaped ``[1, T, H, 512]``.
    """

    _require_sm80(q_nope.device)
    if d_v != _NOPE_DIM:
        raise ValueError(f"sparse_mla_decode_sm80 expects d_v=512, got {d_v}")
    if q_nope.ndim != 3 or q_rope.ndim != 3:
        raise ValueError("sparse_mla_decode_sm80 expects q_nope/q_rope [T, H, D]")
    if q_nope.dtype != torch.bfloat16 or q_rope.dtype != torch.bfloat16:
        raise ValueError("sparse_mla_decode_sm80 expects BF16 queries")
    if (
        q_nope.shape[-1] != _NOPE_DIM
        or q_rope.shape != (*q_nope.shape[:2], _ROPE_DIM)
        or q_nope.shape[1] == 0
    ):
        raise ValueError(
            "SM80 sparse-MLA requires matching nonempty heads and query dimensions 512/64."
        )
    if kv.ndim != 3 or kv.shape[1] != 1 or kv.shape[2] != _KV_DIM:
        raise ValueError(
            f"sparse_mla_decode_sm80 expects kv [N, 1, 576], got {tuple(kv.shape)}"
        )
    if kv.dtype != torch.bfloat16:
        raise ValueError("sparse_mla_decode_sm80 expects a BF16 KV pool")
    if kv.shape[0] == 0:
        raise ValueError("SM80 sparse-MLA requires a nonempty KV pool.")
    if (
        indices.ndim != 3
        or indices.shape[:2] != (q_nope.shape[0], 1)
        or indices.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("SM80 sparse-MLA requires INT32/INT64 indices shaped [T,1,K].")
    for tensor in (q_nope, q_rope, kv, indices):
        if tensor.device != q_nope.device or tensor.stride(-1) != 1:
            raise ValueError(
                "SM80 sparse-MLA requires matching devices and unit dimension strides."
            )
    if (
        not isinstance(sm_scale, (int, float))
        or isinstance(sm_scale, bool)
        or not math.isfinite(sm_scale)
        or sm_scale <= 0
    ):
        raise ValueError("SM80 sparse-MLA requires a finite positive softmax scale.")
    if _verification_fp64 and torch.cuda.is_current_stream_capturing():
        raise ValueError("SM80 sparse-MLA FP64 verification is eager-only.")
    if _verification_precast is not None and (
        not _verification_fp64
        or _verification_precast.shape != (1, *q_nope.shape[:2], _NOPE_DIM)
        or _verification_precast.dtype != torch.float64
        or _verification_precast.device != q_nope.device
        or not _verification_precast.is_contiguous()
    ):
        raise ValueError(
            "Pre-cast capture requires FP64 verification and a contiguous FP64 sink with the output shape/device."
        )

    tokens, heads, _ = q_nope.shape
    indices_2d = indices[:, 0, :]
    topk = indices_2d.shape[1]
    device = q_nope.device

    out = torch.zeros(1, tokens, heads, _NOPE_DIM, dtype=torch.bfloat16, device=device)
    if tokens == 0 or topk == 0:
        if _verification_precast is not None:
            _verification_precast.zero_()
        return out

    head_tiles = triton.cdiv(heads, _HEAD_TILE)
    splits, split_len = _split_k_config(
        tokens,
        heads,
        topk,
        torch.cuda.get_device_properties(device).multi_processor_count,
    )
    partial_dtype = torch.float64 if _verification_fp64 else torch.float32
    partial_acc = torch.empty(
        tokens, splits, heads, _NOPE_DIM, dtype=partial_dtype, device=device
    )
    partial_ml = torch.empty(
        tokens, splits, heads, 2, dtype=partial_dtype, device=device
    )

    _sparse_mla_decode_splitk_kernel[(tokens, head_tiles, splits)](
        q_nope,
        q_rope,
        kv,
        indices_2d,
        partial_acc,
        partial_ml,
        sm_scale,
        kv.shape[0],
        q_nope.stride(0),
        q_nope.stride(1),
        q_rope.stride(0),
        q_rope.stride(1),
        kv.stride(0),
        indices_2d.stride(0),
        partial_acc.stride(0),
        partial_acc.stride(1),
        partial_acc.stride(2),
        partial_ml.stride(0),
        partial_ml.stride(1),
        NUM_HEADS=heads,
        TOPK=topk,
        SPLIT_LEN=split_len,
        HEAD_TILE=_HEAD_TILE,
        KV_CHUNK=_KV_CHUNK,
        NOPE_DIM=_NOPE_DIM,
        ROPE_DIM=_ROPE_DIM,
        VERIFY_FP64=_verification_fp64,
        enable_fp_fusion=not _verification_fp64,
        num_warps=4,
        num_stages=2,
    )
    _sparse_mla_decode_combine_kernel[(tokens, head_tiles)](
        partial_acc,
        partial_ml,
        out,
        out if _verification_precast is None else _verification_precast,
        partial_acc.stride(0),
        partial_acc.stride(1),
        partial_acc.stride(2),
        partial_ml.stride(0),
        partial_ml.stride(1),
        out.stride(1),
        out.stride(2),
        NUM_HEADS=heads,
        NUM_SPLITS=splits,
        HEAD_TILE=_HEAD_TILE,
        NOPE_DIM=_NOPE_DIM,
        VERIFY_FP64=_verification_fp64,
        CAPTURE_PRECAST=_verification_precast is not None,
        enable_fp_fusion=not _verification_fp64,
        num_warps=4,
        num_stages=1,
    )
    return out


__all__ = ["sparse_mla_decode_sm80", "validate_sm80_sparse_mla_config"]
