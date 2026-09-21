from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@functools.cache
def _jit_sparse_module(
    item_size_bytes: int,
    block_size: int,
    num_top_k: int,
    hot_buffer_size: int,
    is_mla: bool = False,
    is_dsv4_layout: bool = False,
    record_miss_plan: bool = False,
    plan_only: bool = False,
    skip_io: bool = False,
) -> Module:
    # Plan-only placement has its own exported binding. skip_io remains a
    # debug-only timing probe and is never used to implement that API.
    template_args = make_cpp_args(
        block_size,
        num_top_k,
        hot_buffer_size,
        is_mla,
        is_dsv4_layout,
        record_miss_plan,
        plan_only,
        skip_io,
    )
    cache_args = make_cpp_args(
        item_size_bytes,
        block_size,
        num_top_k,
        hot_buffer_size,
        is_mla,
        is_dsv4_layout,
        record_miss_plan,
        plan_only,
        skip_io,
    )
    wrapper_name = (
        "plan_cache_to_device_buffer" if plan_only else "load_cache_to_device_buffer"
    )
    return load_jit(
        "sparse_cache",
        *cache_args,
        cuda_files=["hisparse.cuh"],
        cuda_wrappers=[
            (
                wrapper_name,
                f"load_cache_to_device_buffer<{template_args}>",
            )
        ],
    )


@functools.cache
def _jit_copy_planned_module(
    block_size: int,
    is_mla: bool,
    is_dsv4_layout: bool,
    skip_io: bool,
) -> Module:
    template_args = make_cpp_args(block_size, is_mla, is_dsv4_layout, skip_io)
    return load_jit(
        "sparse_copy_planned",
        block_size,
        is_mla,
        is_dsv4_layout,
        skip_io,
        cuda_files=["hisparse.cuh"],
        cuda_wrappers=[
            (
                "copy_cache_planned",
                f"copy_cache_planned<{template_args}>",
            )
        ],
    )


@functools.cache
def _jit_prediction_staging_module(block_size: int) -> Module:
    return load_jit(
        "sparse_prediction_staging",
        block_size,
        cuda_files=["hisparse.cuh"],
        cuda_wrappers=[
            (
                "plan_prediction_staging",
                f"plan_prediction_staging<{block_size}>",
            ),
            (
                "resolve_prediction_staging",
                f"resolve_prediction_staging<{block_size}>",
            ),
        ],
    )


@functools.cache
def _jit_dsv4_transfer_module(block_size: int) -> Module:
    template_args = make_cpp_args(block_size)
    return load_jit(
        "sparse_cache_dsv4_transfer",
        block_size,
        cuda_files=["hisparse.cuh"],
        cuda_wrappers=[
            (
                "transfer_cache_dsv4_mla",
                f"transfer_cache_dsv4_mla<{template_args}>",
            )
        ],
    )


def transfer_cache_dsv4_mla(
    src_ptrs: torch.Tensor,
    dst_ptrs: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    block_size: int = 1024,
) -> None:
    """Transfer DSv4 C4 tokens between page-padded C4 buffers."""
    module = _jit_dsv4_transfer_module(block_size)
    module.transfer_cache_dsv4_mla(
        src_ptrs,
        dst_ptrs,
        src_indices,
        dst_indices,
    )


def _load_cache_to_device_buffer_mla(
    *,
    is_dsv4_layout: bool,
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    page_size: int,
    block_size: int,
    num_real_reqs: torch.Tensor | None,
    miss_src: torch.Tensor | None,
    miss_dst: torch.Tensor | None,
    miss_count: torch.Tensor | None,
    skip_io: bool,
) -> None:
    assert hot_buffer_size >= num_top_k, (
        f"hot_buffer_size ({hot_buffer_size}) must be >= num_top_k ({num_top_k})"
    )

    record_miss_plan = miss_src is not None
    module = _jit_sparse_module(
        item_size_bytes,
        block_size,
        num_top_k,
        hot_buffer_size,
        is_mla=True,
        is_dsv4_layout=is_dsv4_layout,
        record_miss_plan=record_miss_plan,
        plan_only=False,
        skip_io=skip_io,
    )

    empty = torch.empty(0)

    if num_real_reqs is None:
        num_real_reqs = torch.tensor(
            [top_k_tokens.size(0)], dtype=torch.int32, device=top_k_tokens.device
        )

    if record_miss_plan:
        assert miss_dst is not None and miss_count is not None
        assert miss_src.dtype == torch.int64 and miss_dst.dtype == torch.int32
        assert miss_count.dtype == torch.int32
        # The kernel indexes both plan rows with one stride.
        assert miss_src.stride(0) == miss_dst.stride(0)
    else:
        # Unused sentinels; the RecordMissPlan=false instantiation never reads them.
        miss_src = miss_dst = miss_count = empty

    module.load_cache_to_device_buffer(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache,
        empty,
        device_buffer,
        empty,
        top_k_device_locs,
        req_pool_indices,
        seq_lens,
        lru_slots,
        num_real_reqs,
        page_size,
        item_size_bytes,
        miss_src,
        miss_dst,
        miss_count,
    )


def plan_cache_to_device_buffer_mla(
    *,
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    miss_src: torch.Tensor,
    miss_dst: torch.Tensor,
    miss_count: torch.Tensor,
    num_top_k: int,
    hot_buffer_size: int,
    block_size: int = 256,
    num_real_reqs: torch.Tensor | None = None,
) -> None:
    """Place MLA rows and record misses without moving KV bytes.

    The returned locations and mutated placement metadata are pending until
    :func:`copy_cache_planned_mla` has replayed the recorded misses.

    Parameters
    ----------
    top_k_tokens
        Authoritative logical token positions with shape ``[batch, num_top_k]``.
    device_buffer_tokens
        Per-request token tags for the native hot buffer.
    host_cache_locs
        Per-request logical-token to host-row mapping.
    device_buffer_locs
        Per-request slot to physical-device-row mapping.
    top_k_device_locs
        Output physical rows with the same shape as ``top_k_tokens``.
    req_pool_indices
        Native request-pool row for each batch row.
    seq_lens
        Sequence length for each batch row.
    lru_slots
        Per-request hot-slot LRU order.
    miss_src, miss_dst, miss_count
        Output host rows, device rows and valid miss count for each batch row.
    num_top_k
        Compile-time top-k width; must match ``top_k_tokens.shape[1]``.
    hot_buffer_size
        Compile-time number of evictable native slots.
    block_size
        CUDA thread-block size.
    num_real_reqs
        Device scalar containing the active row count. Defaults to the batch size.
    """
    if top_k_tokens.ndim != 2:
        raise ValueError("top_k_tokens must have shape [batch, num_top_k]")
    batch_size, actual_top_k = top_k_tokens.shape
    if actual_top_k != num_top_k:
        raise ValueError(
            f"num_top_k ({num_top_k}) must match top_k_tokens.shape[1] ({actual_top_k})"
        )
    if hot_buffer_size < num_top_k:
        raise ValueError(
            f"hot_buffer_size ({hot_buffer_size}) must be >= num_top_k ({num_top_k})"
        )

    expected_plan_shape = (batch_size, num_top_k)
    if top_k_device_locs.shape != top_k_tokens.shape:
        raise ValueError("top_k_device_locs must match top_k_tokens shape")
    if miss_src.shape != expected_plan_shape or miss_dst.shape != expected_plan_shape:
        raise ValueError(f"miss_src and miss_dst must have shape {expected_plan_shape}")
    if miss_count.shape != (batch_size,):
        raise ValueError(f"miss_count must have shape ({batch_size},)")
    if seq_lens.shape != (batch_size,) or req_pool_indices.shape != (batch_size,):
        raise ValueError("seq_lens and req_pool_indices must have shape [batch]")
    if (
        device_buffer_tokens.ndim != 2
        or device_buffer_tokens.shape[1] < hot_buffer_size
    ):
        raise ValueError("device_buffer_tokens does not cover the hot buffer")
    if device_buffer_locs.shape != device_buffer_tokens.shape:
        raise ValueError("device_buffer_locs must match device_buffer_tokens shape")
    if host_cache_locs.ndim != 2:
        raise ValueError("host_cache_locs must have shape [request, sequence]")
    if lru_slots.ndim != 2 or lru_slots.shape[1] != hot_buffer_size:
        raise ValueError("lru_slots must have shape [request, hot_buffer_size]")

    expected_dtypes = (
        ("top_k_tokens", top_k_tokens, torch.int32),
        ("device_buffer_tokens", device_buffer_tokens, torch.int32),
        ("host_cache_locs", host_cache_locs, torch.int64),
        ("device_buffer_locs", device_buffer_locs, torch.int32),
        ("top_k_device_locs", top_k_device_locs, torch.int32),
        ("lru_slots", lru_slots, torch.int16),
        ("miss_src", miss_src, torch.int64),
        ("miss_dst", miss_dst, torch.int32),
        ("miss_count", miss_count, torch.int32),
    )
    for name, tensor, dtype in expected_dtypes:
        if tensor.dtype != dtype:
            raise ValueError(f"{name} must have dtype {dtype}")
    for name, tensor in (
        ("seq_lens", seq_lens),
        ("req_pool_indices", req_pool_indices),
    ):
        if tensor.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must have dtype torch.int32 or torch.int64")

    tensors = (
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        top_k_device_locs,
        req_pool_indices,
        seq_lens,
        lru_slots,
        miss_src,
        miss_dst,
        miss_count,
    )
    if any(tensor.device != top_k_tokens.device for tensor in tensors):
        raise ValueError("all placement and plan tensors must be on one device")
    if miss_src.stride(1) != 1 or miss_dst.stride(1) != 1:
        raise ValueError("miss plan rows must be contiguous")
    if miss_src.stride(0) != miss_dst.stride(0):
        raise ValueError("miss_src and miss_dst row strides must match")

    if num_real_reqs is None:
        num_real_reqs = torch.tensor(
            [batch_size], dtype=torch.int32, device=top_k_tokens.device
        )
    elif (
        num_real_reqs.dtype != torch.int32
        or num_real_reqs.shape != (1,)
        or num_real_reqs.device != top_k_tokens.device
    ):
        raise ValueError("num_real_reqs must be an int32 device tensor with shape [1]")

    module = _jit_sparse_module(
        item_size_bytes=0,
        block_size=block_size,
        num_top_k=num_top_k,
        hot_buffer_size=hot_buffer_size,
        is_mla=True,
        is_dsv4_layout=False,
        record_miss_plan=True,
        plan_only=True,
        skip_io=False,
    )
    empty = torch.empty(0)
    module.plan_cache_to_device_buffer(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        empty,
        empty,
        empty,
        empty,
        top_k_device_locs,
        req_pool_indices,
        seq_lens,
        lru_slots,
        num_real_reqs,
        1,
        0,
        miss_src,
        miss_dst,
        miss_count,
    )


def load_cache_to_device_buffer_mla(
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    page_size: int = 1,
    block_size: int = 256,
    num_real_reqs: torch.Tensor | None = None,
    miss_src: torch.Tensor | None = None,
    miss_dst: torch.Tensor | None = None,
    miss_count: torch.Tensor | None = None,
    skip_io: bool = False,
) -> None:
    """Generic MLA hisparse swap-in: device + host both linear (stride=item_size_bytes).

    Optional miss_src/miss_dst/miss_count record the miss plan for replay by
    copy_cache_planned_mla; skip_io elides only the KV bytes (timing probe).
    """
    _load_cache_to_device_buffer_mla(
        is_dsv4_layout=False,
        top_k_tokens=top_k_tokens,
        device_buffer_tokens=device_buffer_tokens,
        host_cache_locs=host_cache_locs,
        device_buffer_locs=device_buffer_locs,
        host_cache=host_cache,
        device_buffer=device_buffer,
        top_k_device_locs=top_k_device_locs,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        lru_slots=lru_slots,
        item_size_bytes=item_size_bytes,
        num_top_k=num_top_k,
        hot_buffer_size=hot_buffer_size,
        page_size=page_size,
        block_size=block_size,
        num_real_reqs=num_real_reqs,
        miss_src=miss_src,
        miss_dst=miss_dst,
        miss_count=miss_count,
        skip_io=skip_io,
    )


def copy_cache_planned_mla(
    *,
    miss_src: torch.Tensor,
    miss_dst: torch.Tensor,
    miss_count: torch.Tensor,
    num_real_reqs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    item_size_bytes: int,
    num_blocks: int = 4,
    block_size: int = 1024,
    is_dsv4_layout: bool = False,
    skip_io: bool = False,
) -> None:
    """Replay a recorded miss plan (host_cache -> device_buffer) for a skip layer.

    IO-only, no planning; the small fixed grid keeps the SM footprint low while
    overlapped on a side stream. The anchor's slot table stays valid (lockstep).
    """
    assert miss_src.dtype == torch.int64 and miss_dst.dtype == torch.int32
    assert miss_count.dtype == torch.int32
    module = _jit_copy_planned_module(block_size, True, is_dsv4_layout, skip_io)
    empty = torch.empty(0)
    module.copy_cache_planned(
        miss_src,
        miss_dst,
        miss_count,
        num_real_reqs,
        host_cache,
        empty,
        device_buffer,
        empty,
        num_blocks,
        item_size_bytes,
    )


def plan_prediction_staging_mla(
    *,
    logical_ids: torch.Tensor,
    valid_count: torch.Tensor,
    host_cache_locs: torch.Tensor,
    history_limit: int,
    staged_logical_ids: torch.Tensor,
    staged_host_locs: torch.Tensor,
    staged_dst_locs: torch.Tensor,
    staged_count: torch.Tensor,
    eligible_count: torch.Tensor,
    skipped_count: torch.Tensor,
    block_size: int = 256,
) -> None:
    """Build a fixed-capacity, duplicate-free host-to-stage plan on device.

    All outputs are caller-owned fixed tensors. Counts remain on device, so the
    ordinary inference path performs no scalar download or dynamic compaction.
    """
    assert logical_ids.dtype == torch.int32
    assert valid_count.dtype == torch.int32 and valid_count.numel() == 1
    assert host_cache_locs.dtype == torch.int64
    assert staged_logical_ids.dtype == torch.int64
    assert staged_host_locs.dtype == torch.int64
    assert staged_dst_locs.dtype == torch.int32
    assert staged_count.dtype == torch.int32 and staged_count.numel() == 1
    assert eligible_count.dtype == torch.int32 and eligible_count.numel() == 1
    assert skipped_count.dtype == torch.int32 and skipped_count.numel() == 1
    assert 0 <= history_limit <= host_cache_locs.numel()
    assert logical_ids.numel() == staged_logical_ids.numel()
    assert logical_ids.numel() == staged_host_locs.numel()
    assert logical_ids.numel() == staged_dst_locs.numel()
    tensors = (
        logical_ids,
        valid_count,
        host_cache_locs,
        staged_logical_ids,
        staged_host_locs,
        staged_dst_locs,
        staged_count,
        eligible_count,
        skipped_count,
    )
    assert all(tensor.is_contiguous() for tensor in tensors)
    assert all(tensor.device == logical_ids.device for tensor in tensors)
    staged_count.zero_()
    eligible_count.zero_()
    skipped_count.zero_()
    module = _jit_prediction_staging_module(block_size)
    module.plan_prediction_staging(
        logical_ids,
        valid_count,
        host_cache_locs,
        history_limit,
        staged_logical_ids,
        staged_host_locs,
        staged_dst_locs,
        staged_count,
        eligible_count,
        skipped_count,
    )


def resolve_prediction_staging_mla(
    *,
    miss_src: torch.Tensor,
    miss_dst: torch.Tensor,
    miss_count: torch.Tensor,
    staged_host_locs: torch.Tensor,
    staged_count: torch.Tensor,
    promotion_src: torch.Tensor,
    promotion_dst: torch.Tensor,
    promotion_count: torch.Tensor,
    repair_src: torch.Tensor,
    repair_dst: torch.Tensor,
    repair_count: torch.Tensor,
    block_size: int = 256,
) -> None:
    """Split one native miss plan into fixed promotion and repair plans."""
    assert miss_src.dtype == torch.int64 and miss_src.ndim == 2
    assert miss_dst.dtype == torch.int32 and miss_dst.ndim == 2
    assert miss_count.dtype == torch.int32 and miss_count.numel() == 1
    assert staged_host_locs.dtype == torch.int64
    assert staged_count.dtype == torch.int32 and staged_count.numel() == 1
    assert promotion_src.dtype == torch.int64 and promotion_src.ndim == 2
    assert promotion_dst.dtype == torch.int32 and promotion_dst.ndim == 2
    assert promotion_count.dtype == torch.int32 and promotion_count.numel() == 1
    assert repair_src.dtype == torch.int64 and repair_src.ndim == 2
    assert repair_dst.dtype == torch.int32 and repair_dst.ndim == 2
    assert repair_count.dtype == torch.int32 and repair_count.numel() == 1
    capacity = miss_src.shape[1]
    assert miss_dst.shape == miss_src.shape
    assert promotion_src.shape == miss_src.shape
    assert promotion_dst.shape == miss_dst.shape
    assert repair_src.shape == miss_src.shape
    assert repair_dst.shape == miss_dst.shape
    assert staged_host_locs.numel() == capacity
    tensors = (
        miss_src,
        miss_dst,
        miss_count,
        staged_host_locs,
        staged_count,
        promotion_src,
        promotion_dst,
        promotion_count,
        repair_src,
        repair_dst,
        repair_count,
    )
    assert all(tensor.is_contiguous() for tensor in tensors)
    assert all(tensor.device == miss_src.device for tensor in tensors)
    promotion_count.zero_()
    repair_count.zero_()
    module = _jit_prediction_staging_module(block_size)
    module.resolve_prediction_staging(
        miss_src,
        miss_dst,
        miss_count,
        staged_host_locs,
        staged_count,
        promotion_src,
        promotion_dst,
        promotion_count,
        repair_src,
        repair_dst,
        repair_count,
    )


def load_cache_to_device_buffer_dsv4_mla(
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    page_size: int = 1,
    block_size: int = 256,
    num_real_reqs: torch.Tensor | None = None,
    miss_src: torch.Tensor | None = None,
    miss_dst: torch.Tensor | None = None,
    miss_count: torch.Tensor | None = None,
    skip_io: bool = False,
) -> None:
    """DSv4 hisparse swap-in: page-padded device + page-padded host C4 layout."""
    _load_cache_to_device_buffer_mla(
        is_dsv4_layout=True,
        top_k_tokens=top_k_tokens,
        device_buffer_tokens=device_buffer_tokens,
        host_cache_locs=host_cache_locs,
        device_buffer_locs=device_buffer_locs,
        host_cache=host_cache,
        device_buffer=device_buffer,
        top_k_device_locs=top_k_device_locs,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        lru_slots=lru_slots,
        item_size_bytes=item_size_bytes,
        num_top_k=num_top_k,
        hot_buffer_size=hot_buffer_size,
        page_size=page_size,
        block_size=block_size,
        num_real_reqs=num_real_reqs,
        miss_src=miss_src,
        miss_dst=miss_dst,
        miss_count=miss_count,
        skip_io=skip_io,
    )
