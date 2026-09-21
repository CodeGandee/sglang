import sys
from typing import Literal

import pytest
import torch
from sglang.kernels.ops.kvcache.hisparse import (
    copy_cache_planned_mla,
    load_cache_to_device_buffer_dsv4_mla,
    load_cache_to_device_buffer_mla,
    plan_cache_to_device_buffer_mla,
    plan_prediction_staging_mla,
    publish_prediction_staging_ready_mla,
    resolve_prediction_staging_mla,
    transfer_cache_dsv4_mla,
)
from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_amd_ci(est_time=30, stage="stage-b", runner_config="1-gpu-small-amd")
register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or is_npu()
    or is_xpu()
    or not (is_cuda() or is_hip()),
    reason="HiSparse JIT tests require CUDA/ROCm.",
)

DEVICE = "cuda"
DTYPE = torch.float32
KV_DIM = 8
HOT_BUFFER_SIZE = 4
PADDED_BUFFER_SIZE = HOT_BUFFER_SIZE + 1
HOST_CACHE_SIZE = 16
DEVICE_CACHE_SIZE = 16
ITEM_SIZE_BYTES = KV_DIM * torch.empty((), dtype=DTYPE).element_size()
DSV4_PAGE_SIZE = 64
DSV4_VALUE_BYTES = 576
DSV4_SCALE_BYTES = 8
DSV4_ITEM_BYTES = DSV4_VALUE_BYTES + DSV4_SCALE_BYTES
DSV4_PAGE_BYTES = ((DSV4_ITEM_BYTES * DSV4_PAGE_SIZE + 575) // 576) * 576
DSV4_SCALE_OFFSET = DSV4_VALUE_BYTES * DSV4_PAGE_SIZE


def _host_cache() -> torch.Tensor:
    host_cache = torch.empty(
        (HOST_CACHE_SIZE, 1, KV_DIM), dtype=DTYPE, device="cpu", pin_memory=True
    )
    host_cache.copy_(torch.arange(host_cache.numel(), dtype=DTYPE).view_as(host_cache))
    return host_cache


def _dsv4_token_pattern(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    value = (
        (torch.arange(DSV4_VALUE_BYTES, dtype=torch.int16) + seed)
        .remainder(256)
        .to(torch.uint8)
    )
    scale = (
        (torch.arange(DSV4_SCALE_BYTES, dtype=torch.int16) + seed + 17)
        .remainder(256)
        .to(torch.uint8)
    )
    return value, scale


def _write_dsv4_token(cache: torch.Tensor, loc: int, seed: int) -> None:
    page = loc // DSV4_PAGE_SIZE
    offset = loc % DSV4_PAGE_SIZE
    value, scale = _dsv4_token_pattern(seed)
    cache[page, offset * DSV4_VALUE_BYTES : (offset + 1) * DSV4_VALUE_BYTES].copy_(
        value.to(cache.device)
    )
    scale_start = DSV4_SCALE_OFFSET + offset * DSV4_SCALE_BYTES
    cache[page, scale_start : scale_start + DSV4_SCALE_BYTES].copy_(
        scale.to(cache.device)
    )


def _read_dsv4_token(cache: torch.Tensor, loc: int) -> torch.Tensor:
    page = loc // DSV4_PAGE_SIZE
    offset = loc % DSV4_PAGE_SIZE
    value = cache[page, offset * DSV4_VALUE_BYTES : (offset + 1) * DSV4_VALUE_BYTES]
    scale_start = DSV4_SCALE_OFFSET + offset * DSV4_SCALE_BYTES
    scale = cache[page, scale_start : scale_start + DSV4_SCALE_BYTES]
    return torch.cat([value, scale])


def _dsv4_ptrs(cache: torch.Tensor) -> torch.Tensor:
    return torch.tensor([cache.data_ptr()], dtype=torch.uint64, device=DEVICE)


def _run_kernel(
    *,
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    lru_slots: torch.Tensor,
    seq_len: int | None = None,
    seq_lens: torch.Tensor | None = None,
    seq_lens_dtype: torch.dtype = torch.int32,
    req_pool_indices: torch.Tensor | None = None,
    num_real_reqs: int | None = None,
    output_fill_value: int = -1,
) -> torch.Tensor:
    batch_size = top_k_tokens.shape[0]
    if req_pool_indices is None:
        req_pool_indices = torch.arange(batch_size, dtype=torch.int64, device=DEVICE)
    if seq_lens is None:
        seq_lens = torch.full(
            (batch_size,), seq_len, dtype=seq_lens_dtype, device=DEVICE
        )
    if num_real_reqs is None:
        num_real_reqs = batch_size

    out = torch.full_like(top_k_tokens, output_fill_value)
    load_cache_to_device_buffer_mla(
        top_k_tokens=top_k_tokens,
        device_buffer_tokens=device_buffer_tokens,
        host_cache_locs=host_cache_locs,
        device_buffer_locs=device_buffer_locs,
        host_cache=host_cache,
        device_buffer=device_buffer,
        top_k_device_locs=out,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        lru_slots=lru_slots,
        item_size_bytes=ITEM_SIZE_BYTES,
        num_top_k=top_k_tokens.shape[1],
        hot_buffer_size=HOT_BUFFER_SIZE,
        page_size=1,
        block_size=256,
        num_real_reqs=torch.tensor([num_real_reqs], dtype=torch.int32, device=DEVICE),
    )
    torch.cuda.synchronize()
    return out


def _make_state(
    device_buffer_locs_rows: list[list[int]],
    device_buffer_tokens_rows: list[list[int]],
    newest_tokens: list[int],
):
    host_cache = _host_cache()
    device_buffer = torch.full(
        (DEVICE_CACHE_SIZE, 1, KV_DIM), -1, dtype=DTYPE, device=DEVICE
    )
    device_buffer_locs = torch.tensor(
        device_buffer_locs_rows, dtype=torch.int32, device=DEVICE
    )
    device_buffer_tokens = torch.tensor(
        device_buffer_tokens_rows, dtype=torch.int32, device=DEVICE
    )
    lru_slots = (
        torch.arange(HOT_BUFFER_SIZE, dtype=torch.int16, device=DEVICE)
        .view(1, -1)
        .repeat(device_buffer_locs.shape[0], 1)
    )
    host_cache_locs = (
        torch.arange(HOST_CACHE_SIZE, dtype=torch.int64, device=DEVICE)
        .view(1, -1)
        .repeat(device_buffer_locs.shape[0], 1)
    )

    # Slots 0..3 participate in LRU; slot 4 is the reserved newest slot.
    for rid, newest_token in enumerate(newest_tokens):
        for slot, token in enumerate(device_buffer_tokens_rows[rid][:HOT_BUFFER_SIZE]):
            if token >= 0:
                device_buffer[device_buffer_locs[rid, slot]].copy_(
                    host_cache[token].to(DEVICE, non_blocking=True)
                )
        device_buffer[device_buffer_locs[rid, HOT_BUFFER_SIZE]].copy_(
            host_cache[newest_token].to(DEVICE, non_blocking=True)
        )
    torch.cuda.synchronize()

    return {
        "host_cache": host_cache,
        "device_buffer": device_buffer,
        "device_buffer_locs": device_buffer_locs,
        "device_buffer_tokens": device_buffer_tokens,
        "lru_slots": lru_slots,
        "host_cache_locs": host_cache_locs,
    }


def _make_plan(
    batch_size: int, num_top_k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.full((batch_size, num_top_k), -1, dtype=torch.int64, device=DEVICE),
        torch.full((batch_size, num_top_k), -1, dtype=torch.int32, device=DEVICE),
        torch.full((batch_size,), -1, dtype=torch.int32, device=DEVICE),
    )


@pytest.mark.skipif(is_hip(), reason="DSV4 paged-layout HiSparse test is CUDA-only.")
def test_transfer_cache_dsv4_mla_copies_paged_token() -> None:
    src_cache = torch.zeros((2, DSV4_PAGE_BYTES), dtype=torch.uint8, device=DEVICE)
    dst_cache = torch.zeros(
        (2, DSV4_PAGE_BYTES), dtype=torch.uint8, device="cpu", pin_memory=True
    )
    src_loc = DSV4_PAGE_SIZE + 6
    dst_loc = DSV4_PAGE_SIZE + 1
    _write_dsv4_token(src_cache, src_loc, seed=41)

    transfer_cache_dsv4_mla(
        src_ptrs=_dsv4_ptrs(src_cache),
        dst_ptrs=_dsv4_ptrs(dst_cache),
        src_indices=torch.tensor([src_loc], dtype=torch.int64, device=DEVICE),
        dst_indices=torch.tensor([dst_loc], dtype=torch.int64, device=DEVICE),
    )
    torch.cuda.synchronize()

    assert torch.equal(
        _read_dsv4_token(dst_cache, dst_loc).to(DEVICE),
        _read_dsv4_token(src_cache, src_loc),
    )


@pytest.mark.skipif(is_hip(), reason="DSV4 paged-layout HiSparse test is CUDA-only.")
def test_dsv4_swap_in_reads_paged_host_layout() -> None:
    host_cache = torch.zeros(
        (2, DSV4_PAGE_BYTES), dtype=torch.uint8, device="cpu", pin_memory=True
    )
    device_buffer = torch.zeros((2, DSV4_PAGE_BYTES), dtype=torch.uint8, device=DEVICE)
    host_loc = DSV4_PAGE_SIZE + 1
    swap_loc = DSV4_PAGE_SIZE + 12
    _write_dsv4_token(host_cache, host_loc, seed=41)

    top_k_tokens = torch.tensor([[3]], dtype=torch.int32, device=DEVICE)
    device_buffer_tokens = torch.full(
        (1, PADDED_BUFFER_SIZE), -1, dtype=torch.int32, device=DEVICE
    )
    host_cache_locs = torch.zeros((1, 8), dtype=torch.int64, device=DEVICE)
    host_cache_locs[0, 3] = host_loc
    device_buffer_locs = torch.tensor(
        [[swap_loc, swap_loc + 1, swap_loc + 2, swap_loc + 3, swap_loc + 4]],
        dtype=torch.int32,
        device=DEVICE,
    )
    lru_slots = torch.arange(HOT_BUFFER_SIZE, dtype=torch.int16, device=DEVICE).view(
        1, -1
    )
    out = torch.full_like(top_k_tokens, -1)

    load_cache_to_device_buffer_dsv4_mla(
        top_k_tokens=top_k_tokens,
        device_buffer_tokens=device_buffer_tokens,
        host_cache_locs=host_cache_locs,
        device_buffer_locs=device_buffer_locs,
        host_cache=host_cache,
        device_buffer=device_buffer,
        top_k_device_locs=out,
        req_pool_indices=torch.tensor([0], dtype=torch.int64, device=DEVICE),
        seq_lens=torch.tensor([8], dtype=torch.int32, device=DEVICE),
        lru_slots=lru_slots,
        item_size_bytes=DSV4_ITEM_BYTES,
        num_top_k=1,
        hot_buffer_size=HOT_BUFFER_SIZE,
        page_size=1,
        block_size=256,
        num_real_reqs=torch.tensor([1], dtype=torch.int32, device=DEVICE),
    )
    torch.cuda.synchronize()

    assert out.item() == swap_loc
    assert torch.equal(
        _read_dsv4_token(device_buffer, swap_loc),
        _read_dsv4_token(host_cache, host_loc).to(DEVICE),
    )


def _long_case():
    # One-request baseline used by the stateful cases below:
    # req 0 LRU slots      : [0, 1, 2, 3]
    # req 0 cached tokens  : slot0->1, slot1->4, slot2->2, slot3->5
    # req 0 physical locs  : slot0->9, slot1->7, slot2->3, slot3->5
    # req 0 newest slot    : slot4/newest -> token 7 at physical loc 11
    return _make_state([[9, 7, 3, 5, 11]], [[1, 4, 2, 5, -1]], [7])


@pytest.mark.parametrize("seq_lens_dtype", [torch.int32, torch.int64])
def test_load_cache_to_device_buffer_fast_path(seq_lens_dtype: torch.dtype) -> None:
    host_cache = _host_cache()
    device_buffer = torch.arange(
        DEVICE_CACHE_SIZE * KV_DIM, dtype=DTYPE, device=DEVICE
    ).view(DEVICE_CACHE_SIZE, 1, KV_DIM)
    device_buffer_before = device_buffer.clone()
    device_buffer_locs = torch.tensor(
        [[13, 9, 5, 1, 15]], dtype=torch.int32, device=DEVICE
    )
    device_buffer_tokens = torch.tensor(
        [[10, 11, 12, 13, -1]], dtype=torch.int32, device=DEVICE
    )
    device_buffer_tokens_before = device_buffer_tokens.clone()
    lru_slots = torch.tensor([[0, 1, 2, 3]], dtype=torch.int16, device=DEVICE)
    lru_slots_before = lru_slots.clone()

    # Short-sequence layout:
    # token position 0 -> physical loc 13
    # token position 1 -> physical loc 9
    # token position 2 -> physical loc 5
    #
    # seq_len <= HOT_BUFFER_SIZE should skip host loads and LRU mutations,
    # so top_k_tokens acts like direct indexing into device_buffer_locs.
    out = _run_kernel(
        top_k_tokens=torch.tensor([[2, 0, 1]], dtype=torch.int32, device=DEVICE),
        device_buffer_tokens=device_buffer_tokens,
        host_cache_locs=torch.arange(
            HOST_CACHE_SIZE, dtype=torch.int64, device=DEVICE
        ).view(1, -1),
        device_buffer_locs=device_buffer_locs,
        host_cache=host_cache,
        device_buffer=device_buffer,
        lru_slots=lru_slots,
        seq_len=3,
        seq_lens_dtype=seq_lens_dtype,
    )

    assert torch.equal(out.cpu(), torch.tensor([[5, 13, 9]], dtype=torch.int32))
    assert torch.equal(device_buffer_tokens.cpu(), device_buffer_tokens_before.cpu())
    assert torch.equal(lru_slots.cpu(), lru_slots_before.cpu())
    assert torch.equal(device_buffer.cpu(), device_buffer_before.cpu())


def test_load_cache_to_device_buffer_fast_path_overwrites_stale_output() -> None:
    state = _make_state([[9, 7, 3, 5, 11]], [[0, 1, 2, 3, -1]], [4])

    out = _run_kernel(
        top_k_tokens=torch.tensor([[1, -1, 0, 0]], dtype=torch.int32, device=DEVICE),
        seq_len=2,
        output_fill_value=123456,
        **state,
    )

    assert torch.equal(out.cpu(), torch.tensor([[7, -1, -1, -1]], dtype=torch.int32))


def test_load_cache_to_device_buffer_hits_newest_and_updates_lru() -> None:
    state = _long_case()

    # Query [4, 2, 7]:
    # 4 hits slot1 -> loc 7
    # 2 hits slot2 -> loc 3
    # 7 is the newest token -> reserved newest loc 11
    #
    # Hits move to the MRU tail, so [0, 1, 2, 3] becomes [0, 3, 1, 2].
    out = _run_kernel(
        top_k_tokens=torch.tensor([[4, 2, 7]], dtype=torch.int32, device=DEVICE),
        seq_len=8,
        **state,
    )

    assert torch.equal(out.cpu(), torch.tensor([[7, 3, 11]], dtype=torch.int32))
    assert torch.equal(
        state["device_buffer_tokens"].cpu(),
        torch.tensor([[1, 4, 2, 5, -1]], dtype=torch.int32),
    )
    assert torch.equal(
        state["lru_slots"].cpu(), torch.tensor([[0, 3, 1, 2]], dtype=torch.int16)
    )


def test_load_cache_to_device_buffer_miss_uses_updated_lru_slot() -> None:
    state = _long_case()

    # Step 1: touch tokens [4, 2], so LRU becomes [0, 3, 1, 2].
    # Step 2: query token 6, which is a miss.
    # The kernel should reuse the new LRU head slot0, whose physical loc is 9.
    # This round has no regular hits, so the freshly loaded miss slot ends up at the tail.
    _run_kernel(
        top_k_tokens=torch.tensor([[4, 2]], dtype=torch.int32, device=DEVICE),
        seq_len=8,
        **state,
    )
    out = _run_kernel(
        top_k_tokens=torch.tensor([[6]], dtype=torch.int32, device=DEVICE),
        seq_len=8,
        **state,
    )

    assert torch.equal(out.cpu(), torch.tensor([[9]], dtype=torch.int32))
    assert torch.equal(
        state["device_buffer_tokens"].cpu(),
        torch.tensor([[6, 4, 2, 5, -1]], dtype=torch.int32),
    )
    assert torch.equal(
        state["lru_slots"].cpu(), torch.tensor([[3, 1, 2, 0]], dtype=torch.int16)
    )
    assert torch.equal(state["device_buffer"][9].cpu(), state["host_cache"][6])


@pytest.mark.skipif(
    not is_hip(),
    reason="CUDA transfer_item_warp assumes 16B-aligned items with no sub-8B remainder.",
)
@pytest.mark.parametrize(
    "kv_dim,miss_token",
    [
        # Tokens 0..3 are resident, so the queried token must be >= 4 to miss.
        # The destination is always slot 0, so the source offset
        # (miss_token * item size) is what decides the 16B-alignment check.
        (256, 4),  # 1024B, exactly the gate: one 16B step per lane, no remainder
        (257, 4),  # 1028B: wide path + 4B byte tail
        (258, 4),  # 1032B: wide path + one 64-bit word
        (260, 4),  # 1040B: two wide iterations on lane 0
        (257, 5),  # 1028B, source at 5140: unaligned, wide path skipped
        (5, 4),  # 20B: below the gate, 64-bit loop + 4B byte tail
    ],
)
def test_load_cache_to_device_buffer_miss_copy_is_byte_exact(
    kv_dim: int, miss_token: int
) -> None:
    """A miss must copy the item byte-exactly for any item size and alignment.

    Every other ROCm case in this file uses a 32B item, far below the
    WARP_SIZE * 16 wide-copy gate, so none of them reaches the wide path at all,
    let alone the seam between it and the remainder loops. These sizes sit on
    both sides of the gate and cover each remainder shape.
    """
    item_size_bytes = kv_dim * torch.empty((), dtype=DTYPE).element_size()

    host_cache = torch.empty(
        (HOST_CACHE_SIZE, 1, kv_dim), dtype=DTYPE, device="cpu", pin_memory=True
    )
    host_cache.copy_(torch.arange(host_cache.numel(), dtype=DTYPE).view_as(host_cache))
    device_buffer = torch.full(
        (DEVICE_CACHE_SIZE, 1, kv_dim), -1, dtype=DTYPE, device=DEVICE
    )

    # Slots 0..3 hold tokens 0..3; slot 4 is the reserved newest slot.
    device_buffer_locs = torch.tensor(
        [[0, 1, 2, 3, 4]], dtype=torch.int32, device=DEVICE
    )
    device_buffer_tokens = torch.tensor(
        [[0, 1, 2, 3, -1]], dtype=torch.int32, device=DEVICE
    )
    for slot in range(HOT_BUFFER_SIZE):
        device_buffer[slot].copy_(host_cache[slot].to(DEVICE))
    torch.cuda.synchronize()

    top_k_tokens = torch.tensor([[miss_token]], dtype=torch.int32, device=DEVICE)
    out = torch.full_like(top_k_tokens, -1)

    load_cache_to_device_buffer_mla(
        top_k_tokens=top_k_tokens,
        device_buffer_tokens=device_buffer_tokens,
        host_cache_locs=torch.arange(
            HOST_CACHE_SIZE, dtype=torch.int64, device=DEVICE
        ).view(1, -1),
        device_buffer_locs=device_buffer_locs,
        host_cache=host_cache,
        device_buffer=device_buffer,
        top_k_device_locs=out,
        req_pool_indices=torch.arange(1, dtype=torch.int64, device=DEVICE),
        seq_lens=torch.full((1,), 8, dtype=torch.int32, device=DEVICE),
        lru_slots=torch.arange(HOT_BUFFER_SIZE, dtype=torch.int16, device=DEVICE).view(
            1, -1
        ),
        item_size_bytes=item_size_bytes,
        num_top_k=1,
        hot_buffer_size=HOT_BUFFER_SIZE,
        page_size=1,
        block_size=256,
        num_real_reqs=torch.tensor([1], dtype=torch.int32, device=DEVICE),
    )
    torch.cuda.synchronize()

    # The miss evicts the LRU head (slot 0, physical loc 0) and lands there.
    assert torch.equal(out.cpu(), torch.tensor([[0]], dtype=torch.int32))
    assert torch.equal(device_buffer[0].cpu(), host_cache[miss_token])
    # Neighbouring slots must not be corrupted by an over-copy.
    for slot in range(1, HOT_BUFFER_SIZE):
        assert torch.equal(device_buffer[slot].cpu(), host_cache[slot])


def test_load_cache_to_device_buffer_multiple_misses_copy_all_slots() -> None:
    state = _make_state(
        [[9, 7, 3, 5, 11]],
        [[0, 1, 2, 3, -1]],
        [8],
    )

    out = _run_kernel(
        top_k_tokens=torch.tensor([[4, 5, 6, 7]], dtype=torch.int32, device=DEVICE),
        seq_len=9,
        **state,
    )

    assert torch.equal(out.cpu(), torch.tensor([[9, 7, 3, 5]], dtype=torch.int32))
    assert torch.equal(
        state["device_buffer_tokens"].cpu(),
        torch.tensor([[4, 5, 6, 7, -1]], dtype=torch.int32),
    )
    assert torch.equal(
        state["lru_slots"].cpu(), torch.tensor([[0, 1, 2, 3]], dtype=torch.int16)
    )
    for token, loc in zip([4, 5, 6, 7], [9, 7, 3, 5]):
        assert torch.equal(
            state["device_buffer"][loc].cpu(), state["host_cache"][token]
        )


def test_load_cache_to_device_buffer_batched_with_padding() -> None:
    state = _make_state(
        [
            [9, 7, 3, 5, 11],
            [12, 10, 8, 6, 14],
            [15, 4, 2, 1, 13],
        ],
        [
            [1, 4, 2, 5, -1],
            [0, 1, 2, 3, -1],
            [9, 8, 7, 6, -1],
        ],
        [7, 4, 5],
    )
    padded_tokens_before = state["device_buffer_tokens"][2].clone()
    padded_lru_before = state["lru_slots"][2].clone()

    # req 0: long path
    #   cached tokens/locs : 1@9, 4@7, 2@3, 5@5, newest 7@11
    #   query [4, 6, 7]    : hit loc 7, miss into slot0/loc 9, newest loc 11
    #   LRU update         : remaining evictables [2, 3], then miss [0], then hit [1]
    #                      : [0, 1, 2, 3] -> [2, 3, 0, 1]
    #
    # req 1: fast path
    #   seq_len = 3 <= HOT_BUFFER_SIZE, so [2, 1, 0] maps directly to locs [8, 10, 12]
    #
    # req 2: padded block
    #   num_real_reqs = 2 means this row must be ignored entirely.
    out = _run_kernel(
        top_k_tokens=torch.tensor(
            [[4, 6, 7], [2, 1, 0], [9, 8, 7]], dtype=torch.int32, device=DEVICE
        ),
        seq_lens=torch.tensor([8, 3, 8], dtype=torch.int32, device=DEVICE),
        num_real_reqs=2,
        output_fill_value=123456,
        **state,
    )

    assert torch.equal(
        out.cpu(),
        torch.tensor([[7, 9, 11], [8, 10, 12], [-1, -1, -1]], dtype=torch.int32),
    )
    assert torch.equal(
        state["device_buffer_tokens"][:2].cpu(),
        torch.tensor([[6, 4, 2, 5, -1], [0, 1, 2, 3, -1]], dtype=torch.int32),
    )
    assert torch.equal(
        state["lru_slots"][:2].cpu(),
        torch.tensor([[2, 3, 0, 1], [0, 1, 2, 3]], dtype=torch.int16),
    )
    assert torch.equal(
        state["device_buffer_tokens"][2].cpu(), padded_tokens_before.cpu()
    )
    assert torch.equal(state["lru_slots"][2].cpu(), padded_lru_before.cpu())
    assert torch.equal(state["device_buffer"][9].cpu(), state["host_cache"][6])


@pytest.mark.skipif(is_hip(), reason="M4.2 qualifies the CUDA GLM MLA profile.")
@pytest.mark.parametrize("prediction_staging", ["off", "mixed", "zero_hit"])
def test_plan_then_copy_matches_fused_bytes_at_glm_profile(
    prediction_staging: Literal["off", "mixed", "zero_hit"],
) -> None:
    """Plan/replay must exactly match fused placement and bytes over two steps."""
    num_top_k = 2048
    hot_buffer_size = 4096
    row_width = 576
    seq_capacity = 8192
    newest_token = seq_capacity - 1
    item_size_bytes = row_width * torch.empty((), dtype=torch.bfloat16).element_size()
    num_requests = 2
    slots_per_request = hot_buffer_size + 1

    token_ids = torch.arange(seq_capacity, dtype=torch.int32)
    byte_columns = torch.arange(row_width, dtype=torch.int32)
    source_rows = (
        ((token_ids[:, None] * 131 + byte_columns[None, :] * 17) % 32749)
        .sub(16384)
        .to(torch.bfloat16)
        .view(seq_capacity, 1, row_width)
    )

    # Request 1 owns odd, permuted host/device rows; request 0 occupies the even
    # guard rows and must remain byte- and metadata-identical throughout.
    host_rows = num_requests * seq_capacity
    host_cache = torch.full(
        (host_rows, 1, row_width),
        -321,
        dtype=torch.bfloat16,
        device="cpu",
        pin_memory=True,
    )
    host_locs_cpu = torch.stack(
        (
            2 * torch.arange(seq_capacity, dtype=torch.int64),
            2
            * ((torch.arange(seq_capacity, dtype=torch.int64) * 37 + 11) % seq_capacity)
            + 1,
        )
    )
    host_cache[host_locs_cpu[1]] = source_rows
    host_before = host_cache.clone()

    slot_ids = torch.arange(slots_per_request, dtype=torch.int64)
    device_locs_cpu = torch.stack(
        (
            2 * slot_ids,
            2 * ((slot_ids * 31 + 7) % slots_per_request) + 1,
        )
    ).to(torch.int32)
    device_buffer_tokens_cpu = torch.full(
        (num_requests, slots_per_request), -1, dtype=torch.int32
    )
    device_buffer_tokens_cpu[:, :hot_buffer_size] = torch.arange(
        hot_buffer_size, dtype=torch.int32
    )
    lru_slots_cpu = torch.arange(hot_buffer_size, dtype=torch.int16).repeat(
        num_requests, 1
    )

    device_rows = num_requests * slots_per_request
    initial_device = torch.full(
        (device_rows, 1, row_width),
        -777,
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    initial_device[device_locs_cpu[0].to(DEVICE)] = -555
    request_device_locs = device_locs_cpu[1].to(DEVICE)
    initial_device[request_device_locs[:hot_buffer_size]] = source_rows[
        :hot_buffer_size
    ].to(DEVICE)
    # Both rounds evict from this poisoned prefix. Hits live outside it.
    initial_device[request_device_locs[:num_top_k]] = -777
    initial_device[request_device_locs[hot_buffer_size]] = -source_rows[
        newest_token
    ].to(DEVICE)
    initial_device_before = initial_device.clone()

    host_cache_locs = host_locs_cpu.to(DEVICE)
    # This optional prediction source is unpublished and never real demand.
    host_cache_locs[1, 7001] = -1
    device_buffer_locs = device_locs_cpu.to(DEVICE)
    req_pool_indices = torch.tensor([1], dtype=torch.int64, device=DEVICE)
    seq_lens = torch.tensor([seq_capacity], dtype=torch.int32, device=DEVICE)
    num_real_reqs = torch.tensor([1], dtype=torch.int32, device=DEVICE)

    def make_mutable_state() -> dict[str, torch.Tensor]:
        return {
            "device_buffer_tokens": device_buffer_tokens_cpu.to(DEVICE),
            "lru_slots": lru_slots_cpu.to(DEVICE),
            "device_buffer": initial_device.clone(),
        }

    fused = make_mutable_state()
    split = make_mutable_state()
    request_zero_tokens = device_buffer_tokens_cpu[0].clone()
    request_zero_lru = lru_slots_cpu[0].clone()

    def byte_view(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.contiguous().view(torch.uint8)

    round_inputs = (
        (
            torch.cat(
                (
                    torch.arange(2048, 3072, dtype=torch.int32),
                    torch.arange(4096, 5119, dtype=torch.int32),
                    torch.tensor([newest_token], dtype=torch.int32),
                )
            ),
            torch.arange(4096, 5119, dtype=torch.int32),
        ),
        (
            torch.cat(
                (
                    torch.arange(4096, 5119, dtype=torch.int32),
                    torch.arange(5119, 6143, dtype=torch.int32),
                    torch.tensor([newest_token], dtype=torch.int32),
                )
            ),
            torch.arange(5119, 6143, dtype=torch.int32),
        ),
    )
    permutation = (torch.arange(num_top_k, dtype=torch.int64) * 1031) % num_top_k

    for raw_top_k, miss_tokens in round_inputs:
        top_k_tokens = raw_top_k[permutation].view(1, -1).to(DEVICE)
        if prediction_staging != "off":
            authoritative_before = {
                name: tensor.clone() for name, tensor in split.items()
            }
            host_map_before = host_cache_locs.clone()
            # Predict half the misses, one resident and one unnecessary row.
            # Include duplicates, unavailable history, the newest device-only
            # row and a private future position in the valid prediction prefix.
            predicted = torch.cat(
                (
                    miss_tokens[::2],
                    miss_tokens[:1],
                    torch.tensor([2500, 7000, 7001, newest_token, seq_capacity, -1]),
                )
            ).to(torch.int32)
            if prediction_staging == "zero_hit":
                predicted = torch.tensor(
                    [7000, 7000, 7001, newest_token, seq_capacity, -1],
                    dtype=torch.int32,
                )
            prediction_ids = torch.full(
                (num_top_k,), -1, dtype=torch.int32, device=DEVICE
            )
            prediction_ids[: predicted.numel()].copy_(predicted)
            staged_ids = torch.empty(num_top_k, dtype=torch.int64, device=DEVICE)
            stage_src, stage_dst, stage_count = _make_plan(1, num_top_k)
            eligible_count = torch.zeros(1, dtype=torch.int32, device=DEVICE)
            skipped_count = torch.zeros_like(eligible_count)
            stage = torch.full(
                (num_top_k, 1, row_width), -999, dtype=torch.bfloat16, device=DEVICE
            )
            plan_prediction_staging_mla(
                logical_ids=prediction_ids,
                valid_count=torch.tensor(
                    [predicted.numel()], dtype=torch.int32, device=DEVICE
                ),
                host_cache_locs=host_cache_locs[1],
                history_limit=newest_token,
                staged_logical_ids=staged_ids,
                staged_host_locs=stage_src[0],
                staged_dst_locs=stage_dst[0],
                staged_count=stage_count,
                eligible_count=eligible_count,
                skipped_count=skipped_count,
            )
            copy_cache_planned_mla(
                miss_src=stage_src,
                miss_dst=stage_dst,
                miss_count=stage_count,
                num_real_reqs=num_real_reqs,
                host_cache=host_cache,
                device_buffer=stage,
                item_size_bytes=item_size_bytes,
            )
            torch.cuda.synchronize()
            expected_stage_ids = sorted(
                {
                    token
                    for token in predicted.tolist()
                    if 0 <= token < newest_token and token != 7001
                }
            )
            staged_count = int(stage_count.item())
            actual_stage_ids = staged_ids[:staged_count].cpu()
            assert sorted(actual_stage_ids.tolist()) == expected_stage_ids
            assert int(eligible_count.item()) == predicted.numel()
            assert int(skipped_count.item()) == predicted.numel() - staged_count
            assert torch.equal(
                byte_view(stage[:staged_count].cpu()),
                byte_view(source_rows[actual_stage_ids]),
            )
            for name, previous in authoritative_before.items():
                assert torch.equal(split[name], previous), name
            assert torch.equal(host_cache_locs, host_map_before)
            assert torch.equal(byte_view(host_cache), byte_view(host_before))
        fused_out = torch.full_like(top_k_tokens, -1)
        load_cache_to_device_buffer_mla(
            top_k_tokens=top_k_tokens,
            device_buffer_tokens=fused["device_buffer_tokens"],
            host_cache_locs=host_cache_locs,
            device_buffer_locs=device_buffer_locs,
            host_cache=host_cache,
            device_buffer=fused["device_buffer"],
            top_k_device_locs=fused_out,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            lru_slots=fused["lru_slots"],
            item_size_bytes=item_size_bytes,
            num_top_k=num_top_k,
            hot_buffer_size=hot_buffer_size,
            block_size=960,
            num_real_reqs=num_real_reqs,
        )

        split_before = split["device_buffer"].clone()
        split_out = torch.full_like(top_k_tokens, -1)
        miss_src, miss_dst, miss_count = _make_plan(1, num_top_k)
        plan_cache_to_device_buffer_mla(
            top_k_tokens=top_k_tokens,
            device_buffer_tokens=split["device_buffer_tokens"],
            host_cache_locs=host_cache_locs,
            device_buffer_locs=device_buffer_locs,
            top_k_device_locs=split_out,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            lru_slots=split["lru_slots"],
            miss_src=miss_src,
            miss_dst=miss_dst,
            miss_count=miss_count,
            num_top_k=num_top_k,
            hot_buffer_size=hot_buffer_size,
            block_size=960,
            num_real_reqs=num_real_reqs,
        )
        torch.cuda.synchronize()

        assert torch.equal(split_out, fused_out)
        assert torch.equal(split["device_buffer_tokens"], fused["device_buffer_tokens"])
        assert torch.equal(split["lru_slots"], fused["lru_slots"])
        assert torch.equal(byte_view(split["device_buffer"]), byte_view(split_before))
        assert torch.equal(byte_view(host_cache), byte_view(host_before))

        count = int(miss_count.item())
        assert count == miss_tokens.numel()
        recorded_src = miss_src[0, :count].cpu()
        recorded_dst = miss_dst[0, :count].to(torch.int64)
        expected_src = host_locs_cpu[1, miss_tokens.to(torch.int64)]
        assert torch.equal(recorded_src.sort().values, expected_src.sort().values)
        assert torch.unique(recorded_dst).numel() == count
        assert torch.all(split_before[recorded_dst] == -777)

        independently_replayed = split_before.clone()
        independently_replayed[recorded_dst] = host_cache[recorded_src].to(DEVICE)
        host_only = (
            split_before.clone()
            if prediction_staging != "off"
            else split["device_buffer"]
        )
        copy_cache_planned_mla(
            miss_src=miss_src,
            miss_dst=miss_dst,
            miss_count=miss_count,
            num_real_reqs=num_real_reqs,
            host_cache=host_cache,
            device_buffer=host_only,
            item_size_bytes=item_size_bytes,
        )
        if prediction_staging != "off":
            original_plan = (miss_src.clone(), miss_dst.clone(), miss_count.clone())
            promotion_src, promotion_dst, promotion_count = _make_plan(1, num_top_k)
            repair_src, repair_dst, repair_count = _make_plan(1, num_top_k)
            resolve_prediction_staging_mla(
                miss_src=miss_src,
                miss_dst=miss_dst,
                miss_count=miss_count,
                staged_host_locs=stage_src[0],
                staged_count=stage_count,
                promotion_src=promotion_src,
                promotion_dst=promotion_dst,
                promotion_count=promotion_count,
                repair_src=repair_src,
                repair_dst=repair_dst,
                repair_count=repair_count,
            )
            for src, dst, valid, source in (
                (promotion_src, promotion_dst, promotion_count, stage),
                (repair_src, repair_dst, repair_count, host_cache),
            ):
                copy_cache_planned_mla(
                    miss_src=src,
                    miss_dst=dst,
                    miss_count=valid,
                    num_real_reqs=num_real_reqs,
                    host_cache=source,
                    device_buffer=split["device_buffer"],
                    item_size_bytes=item_size_bytes,
                )
            # Followers always replay every original miss from their own layer.
            follower_host = torch.empty_like(host_cache, pin_memory=True)
            follower_host.copy_(-host_cache)
            follower = -split_before
            copy_cache_planned_mla(
                miss_src=miss_src,
                miss_dst=miss_dst,
                miss_count=miss_count,
                num_real_reqs=num_real_reqs,
                host_cache=follower_host,
                device_buffer=follower,
                item_size_bytes=item_size_bytes,
            )
        torch.cuda.synchronize()

        assert torch.equal(
            byte_view(split["device_buffer"]), byte_view(independently_replayed)
        )
        assert torch.equal(
            byte_view(split["device_buffer"]), byte_view(fused["device_buffer"])
        )
        assert torch.equal(byte_view(split["device_buffer"]), byte_view(host_only))
        selected_tokens = raw_top_k[permutation]
        expected_selected = source_rows[selected_tokens.to(torch.int64)].clone()
        expected_selected[selected_tokens == newest_token] = -source_rows[newest_token]
        actual_selected = split["device_buffer"].index_select(0, split_out.view(-1))
        assert torch.equal(
            byte_view(actual_selected.cpu()), byte_view(expected_selected)
        )
        assert torch.equal(
            byte_view(split["device_buffer"][device_locs_cpu[0].to(DEVICE)]),
            byte_view(initial_device_before[device_locs_cpu[0].to(DEVICE)]),
        )
        assert torch.equal(split["device_buffer_tokens"][0].cpu(), request_zero_tokens)
        assert torch.equal(split["lru_slots"][0].cpu(), request_zero_lru)
        if prediction_staging != "off":
            promoted = int(promotion_count.item())
            repaired = int(repair_count.item())
            assert repaired > 0 and promoted + repaired == count
            if prediction_staging == "mixed":
                assert promoted > 0
            else:
                assert promoted == 0 and repaired == count
            promotion_destinations = promotion_dst[0, :promoted].to(torch.int64)
            repair_destinations = repair_dst[0, :repaired].cpu().tolist()
            promoted_set = set(promotion_destinations.cpu().tolist())
            assert promoted_set.isdisjoint(repair_destinations)
            assert promoted_set | set(repair_destinations) == set(
                recorded_dst.cpu().tolist()
            )
            for actual, before in zip((miss_src, miss_dst, miss_count), original_plan):
                assert torch.equal(actual, before)
            assert torch.equal(byte_view(follower), byte_view(-host_only))
            assert torch.equal(
                byte_view(follower.index_select(0, split_out.view(-1)).cpu()),
                byte_view(-expected_selected),
            )

            if promoted == 0:
                continue

            # A separate source-use probe: completed staged bytes must survive
            # changing only their synthetic host source. Host repair would now
            # return a different pattern, so hidden host-only copying fails.
            saved = (
                split["device_buffer"].index_select(0, promotion_destinations).clone()
            )
            source_positions = promotion_src[0, :promoted]
            altered_host_positions = (
                stage_src[0].index_select(0, source_positions).cpu()
            )
            host_cache[altered_host_positions] = 123
            split["device_buffer"][promotion_destinations] = -777
            copy_cache_planned_mla(
                miss_src=promotion_src,
                miss_dst=promotion_dst,
                miss_count=promotion_count,
                num_real_reqs=num_real_reqs,
                host_cache=stage,
                device_buffer=split["device_buffer"],
                item_size_bytes=item_size_bytes,
            )
            torch.cuda.synchronize()
            assert torch.equal(
                byte_view(
                    split["device_buffer"].index_select(0, promotion_destinations)
                ),
                byte_view(saved),
            )
            assert not torch.equal(
                byte_view(saved.cpu()), byte_view(host_cache[altered_host_positions])
            )
            host_cache.copy_(host_before)


def test_load_cache_to_device_buffer_dsv4_mla_miss_copy_layout() -> None:
    # Both the host cache and the device buffer use the page-padded C4 layout,
    # matching DeepSeekV4PagedHostPool, the backup/write path, and the swap-in
    # kernel on both CUDA and ROCm. The miss copy must read the host source with
    # paged addressing (get_pointer_paged), not a linear per-item stride.
    num_pages = (HOST_CACHE_SIZE + DSV4_PAGE_SIZE - 1) // DSV4_PAGE_SIZE

    state = _long_case()
    host_cache = torch.zeros(
        (num_pages, DSV4_PAGE_BYTES),
        dtype=torch.uint8,
        device="cpu",
        pin_memory=True,
    )
    for token in range(HOST_CACHE_SIZE):
        _write_dsv4_token(host_cache, token, seed=token + 1)

    device_buffer = torch.full(
        (num_pages, DSV4_PAGE_BYTES),
        0xFF,
        dtype=torch.uint8,
        device=DEVICE,
    )
    out = torch.full((1, 1), -1, dtype=torch.int32, device=DEVICE)

    # Token 6 is a miss in _long_case(), so it should be copied into evict slot 0,
    # whose physical device loc is 9.
    load_cache_to_device_buffer_dsv4_mla(
        top_k_tokens=torch.tensor([[6]], dtype=torch.int32, device=DEVICE),
        device_buffer_tokens=state["device_buffer_tokens"],
        host_cache_locs=state["host_cache_locs"],
        device_buffer_locs=state["device_buffer_locs"],
        host_cache=host_cache,
        device_buffer=device_buffer,
        top_k_device_locs=out,
        req_pool_indices=torch.tensor([0], dtype=torch.int64, device=DEVICE),
        seq_lens=torch.tensor([8], dtype=torch.int32, device=DEVICE),
        lru_slots=state["lru_slots"],
        item_size_bytes=DSV4_ITEM_BYTES,
        num_top_k=1,
        hot_buffer_size=HOT_BUFFER_SIZE,
        page_size=DSV4_PAGE_SIZE,
        block_size=256,
        num_real_reqs=torch.tensor([1], dtype=torch.int32, device=DEVICE),
    )
    torch.cuda.synchronize()

    assert torch.equal(out.cpu(), torch.tensor([[9]], dtype=torch.int32))

    # host_cache_locs[token=6] == 6 in _long_case(); evict slot 0 -> device loc 9.
    assert torch.equal(
        _read_dsv4_token(device_buffer, 9).cpu(),
        _read_dsv4_token(host_cache, 6),
    )


@pytest.mark.skipif(
    not is_hip(), reason="Covers the ROCm wavefront64 fused DSv4 token copy."
)
def test_load_cache_to_device_buffer_dsv4_fused_copy_multi_miss() -> None:
    """Several DSv4 misses in one launch must each land byte-exact.

    The fused copy walks the 576B value and the 8B scale as one 73-word space,
    so the seam between them falls on a lane index rather than a call boundary.
    Vary both the source and the destination page offset, including tokens on
    the second page, so the seam is not always at the same address.
    """
    hot_buffer_size = 4
    num_pages = 2
    # seq_len stays above the queried tokens so none of them is the newest
    # token, which the kernel places without a host copy.
    seq_len = 16
    host_locs = list(range(seq_len))
    miss_tokens = [4, 5, 6, 7]
    # Source offsets: mid-page, last slot of page 0, first slot of page 1,
    # last slot of page 1.
    for token, loc in zip(miss_tokens, [10, 63, 64, 127]):
        host_locs[token] = loc
    # Destination offsets: first, second, last of page 0, then page 1.
    device_locs = [0, 1, 63, 64, 65]

    host_cache = torch.zeros(
        (num_pages, DSV4_PAGE_BYTES), dtype=torch.uint8, device="cpu", pin_memory=True
    )
    for loc in host_locs:
        _write_dsv4_token(host_cache, loc, seed=loc + 1)

    device_buffer = torch.full(
        (num_pages, DSV4_PAGE_BYTES), 0xFF, dtype=torch.uint8, device=DEVICE
    )

    top_k_tokens = torch.tensor([miss_tokens], dtype=torch.int32, device=DEVICE)
    out = torch.full_like(top_k_tokens, -1)

    load_cache_to_device_buffer_dsv4_mla(
        top_k_tokens=top_k_tokens,
        device_buffer_tokens=torch.tensor(
            [[0, 1, 2, 3, -1]], dtype=torch.int32, device=DEVICE
        ),
        host_cache_locs=torch.tensor([host_locs], dtype=torch.int64, device=DEVICE),
        device_buffer_locs=torch.tensor(
            [device_locs], dtype=torch.int32, device=DEVICE
        ),
        host_cache=host_cache,
        device_buffer=device_buffer,
        top_k_device_locs=out,
        req_pool_indices=torch.tensor([0], dtype=torch.int64, device=DEVICE),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=DEVICE),
        lru_slots=torch.arange(hot_buffer_size, dtype=torch.int16, device=DEVICE).view(
            1, -1
        ),
        item_size_bytes=DSV4_ITEM_BYTES,
        num_top_k=len(miss_tokens),
        hot_buffer_size=hot_buffer_size,
        page_size=DSV4_PAGE_SIZE,
        block_size=256,
        num_real_reqs=torch.tensor([1], dtype=torch.int32, device=DEVICE),
    )
    torch.cuda.synchronize()

    # Which slot each miss evicts is up to the LRU, so take the destinations
    # from the kernel; only require that they are distinct and in range.
    landed = out.cpu().tolist()[0]
    assert len(set(landed)) == len(landed)
    assert set(landed).issubset(device_locs)

    device_cpu = device_buffer.cpu()
    for token, dst_loc in zip(miss_tokens, landed):
        assert torch.equal(
            _read_dsv4_token(device_cpu, dst_loc),
            _read_dsv4_token(host_cache, host_locs[token]),
        ), f"token {token} -> device loc {dst_loc}"

    # Slots the kernel never wrote must keep their fill, so an over-copy that
    # ran past the value or the scale would be caught.
    for loc in set(device_locs) - set(landed):
        assert torch.all(_read_dsv4_token(device_cpu, loc) == 0xFF)


@pytest.mark.skipif(
    not is_hip(), reason="Covers a ROCm wavefront64 LRU writeback regression."
)
def test_load_cache_to_device_buffer_rocm_large_lru_writeback() -> None:
    top_k = 2048
    hot_buffer_size = 4096
    seq_len = 7299
    kv_dim = 4
    item_size_bytes = kv_dim * torch.empty((), dtype=DTYPE).element_size()

    top_k_tokens = torch.cat(
        [
            torch.arange(1000, 2000, dtype=torch.int32),
            torch.arange(5000, 6048, dtype=torch.int32),
        ]
    ).view(1, -1)
    device_buffer_tokens = torch.arange(hot_buffer_size, dtype=torch.int32).view(1, -1)
    device_buffer_locs = torch.arange(hot_buffer_size + 1, dtype=torch.int32).view(
        1, -1
    )
    lru_slots = torch.arange(hot_buffer_size, dtype=torch.int16).view(1, -1)
    host_cache_locs = torch.arange(seq_len, dtype=torch.int64).view(1, -1)

    top_k_tokens = top_k_tokens.to(DEVICE)
    device_buffer_tokens = device_buffer_tokens.to(DEVICE)
    device_buffer_locs = device_buffer_locs.to(DEVICE)
    lru_slots = lru_slots.to(DEVICE)
    host_cache_locs = host_cache_locs.to(DEVICE)

    host_cache = torch.empty((seq_len, 1, kv_dim), dtype=DTYPE, pin_memory=True)
    host_cache.zero_()
    device_buffer = torch.empty(
        (hot_buffer_size + 1, 1, kv_dim), dtype=DTYPE, device=DEVICE
    )
    out = torch.full_like(top_k_tokens, -1)

    load_cache_to_device_buffer_mla(
        top_k_tokens=top_k_tokens,
        device_buffer_tokens=device_buffer_tokens,
        host_cache_locs=host_cache_locs,
        device_buffer_locs=device_buffer_locs,
        host_cache=host_cache,
        device_buffer=device_buffer,
        top_k_device_locs=out,
        req_pool_indices=torch.tensor([0], dtype=torch.int64, device=DEVICE),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=DEVICE),
        lru_slots=lru_slots,
        item_size_bytes=item_size_bytes,
        num_top_k=top_k,
        hot_buffer_size=hot_buffer_size,
        page_size=1,
        block_size=1024,
        num_real_reqs=torch.tensor([1], dtype=torch.int32, device=DEVICE),
    )
    torch.cuda.synchronize()

    expected_lru = torch.cat(
        [
            torch.arange(2048, 4096, dtype=torch.int16),
            torch.arange(0, 1000, dtype=torch.int16),
            torch.arange(2000, 2048, dtype=torch.int16),
            torch.arange(1000, 2000, dtype=torch.int16),
        ]
    )
    assert torch.equal(lru_slots.cpu().view(-1), expected_lru)


@pytest.mark.skipif(is_hip(), reason="M4.4 ready-tag gate is CUDA-qualified.")
def test_prediction_source_resolver_fails_closed_until_matching_ready_tag() -> None:
    """A stale/unpublished lease tag routes every miss to urgent repair."""
    miss_src = torch.tensor([[11, 22, 33, -1]], dtype=torch.int64, device=DEVICE)
    miss_dst = torch.tensor([[4, 5, 6, -1]], dtype=torch.int32, device=DEVICE)
    miss_count = torch.tensor([3], dtype=torch.int32, device=DEVICE)
    staged_src = torch.tensor([11, 33, -1, -1], dtype=torch.int64, device=DEVICE)
    staged_count = torch.tensor([2], dtype=torch.int32, device=DEVICE)
    ready_tag = torch.zeros(1, dtype=torch.int64, device=DEVICE)
    promotion_src = torch.empty_like(miss_src)
    promotion_dst = torch.empty_like(miss_dst)
    promotion_count = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    repair_src = torch.empty_like(miss_src)
    repair_dst = torch.empty_like(miss_dst)
    repair_count = torch.zeros(1, dtype=torch.int32, device=DEVICE)

    publish_prediction_staging_ready_mla(ready_tag=ready_tag, expected_tag=17)
    resolve_prediction_staging_mla(
        miss_src=miss_src,
        miss_dst=miss_dst,
        miss_count=miss_count,
        staged_host_locs=staged_src,
        staged_count=staged_count,
        ready_tag=ready_tag,
        expected_tag=99,
        promotion_src=promotion_src,
        promotion_dst=promotion_dst,
        promotion_count=promotion_count,
        repair_src=repair_src,
        repair_dst=repair_dst,
        repair_count=repair_count,
    )
    torch.cuda.synchronize()
    assert int(promotion_count.item()) == 0
    assert int(repair_count.item()) == 3

    resolve_prediction_staging_mla(
        miss_src=miss_src,
        miss_dst=miss_dst,
        miss_count=miss_count,
        staged_host_locs=staged_src,
        staged_count=staged_count,
        ready_tag=ready_tag,
        expected_tag=17,
        promotion_src=promotion_src,
        promotion_dst=promotion_dst,
        promotion_count=promotion_count,
        repair_src=repair_src,
        repair_dst=repair_dst,
        repair_count=repair_count,
    )
    torch.cuda.synchronize()
    assert int(promotion_count.item()) == 2
    assert int(repair_count.item()) == 1


@pytest.mark.skipif(is_hip(), reason="M4.4 urgent-independence gate is CUDA-qualified.")
def test_delayed_stage_leases_do_not_block_urgent_repair() -> None:
    """Urgent repair completes while both predictive producers are gated.

    The first real miss overlaps a row that a delayed lease will eventually
    stage; the second miss is absent from both leases.  Neither producer may
    write the authoritative destination, and the urgent result must match the
    host reference before the producer gate is released.
    """

    host_cache = _host_cache()
    item_size_bytes = ITEM_SIZE_BYTES
    real_buffer = torch.full(
        (8, 1, KV_DIM), -777, dtype=DTYPE, device=DEVICE
    )
    num_real_reqs = torch.tensor([1], dtype=torch.int32, device=DEVICE)
    gate = torch.cuda.Event()
    producer_streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    producer_done = [torch.cuda.Event(), torch.cuda.Event()]
    stage_buffers = [
        torch.full((4, 1, KV_DIM), -999, dtype=DTYPE, device=DEVICE),
        torch.full((4, 1, KV_DIM), -999, dtype=DTYPE, device=DEVICE),
    ]
    stage_sources = [
        torch.tensor([[1, -1, -1, -1]], dtype=torch.int64, device=DEVICE),
        torch.tensor([[2, -1, -1, -1]], dtype=torch.int64, device=DEVICE),
    ]
    stage_destinations = [
        torch.tensor([[0, -1, -1, -1]], dtype=torch.int32, device=DEVICE),
        torch.tensor([[0, -1, -1, -1]], dtype=torch.int32, device=DEVICE),
    ]
    stage_counts = [torch.tensor([1], dtype=torch.int32, device=DEVICE) for _ in range(2)]
    ready_tags = [torch.zeros(1, dtype=torch.int64, device=DEVICE) for _ in range(2)]
    expected_tags = (101, 102)
    hold = torch.cuda.Stream()
    with torch.cuda.stream(hold):
        torch.cuda._sleep(10_000_000_000)
        gate.record(hold)

    for stream, done, stage, source, destination, count, ready_tag, expected in zip(
        producer_streams,
        producer_done,
        stage_buffers,
        stage_sources,
        stage_destinations,
        stage_counts,
        ready_tags,
        expected_tags,
    ):
        stream.wait_event(gate)
        with torch.cuda.stream(stream):
            copy_cache_planned_mla(
                miss_src=source,
                miss_dst=destination,
                miss_count=count,
                num_real_reqs=num_real_reqs,
                host_cache=host_cache,
                device_buffer=stage,
                item_size_bytes=item_size_bytes,
            )
            publish_prediction_staging_ready_mla(
                ready_tag=ready_tag,
                expected_tag=expected,
            )
            done.record(stream)

    urgent = torch.cuda.Stream(priority=-1)
    urgent_done = torch.cuda.Event()
    real_misses = (
        torch.tensor([[1, 3, -1, -1]], dtype=torch.int64, device=DEVICE),
        torch.tensor([[2, 3, -1, -1]], dtype=torch.int64, device=DEVICE),
    )
    for miss_src, ready_tag, expected_tag, destination in zip(
        real_misses,
        ready_tags,
        expected_tags,
        (0, 1),
    ):
        miss_dst = torch.tensor(
            [[destination, destination + 2, -1, -1]],
            dtype=torch.int32,
            device=DEVICE,
        )
        miss_count = torch.tensor([2], dtype=torch.int32, device=DEVICE)
        promotion_src, promotion_dst, promotion_count = _make_plan(1, 4)
        repair_src, repair_dst, repair_count = _make_plan(1, 4)
        with torch.cuda.stream(urgent):
            resolve_prediction_staging_mla(
                miss_src=miss_src,
                miss_dst=miss_dst,
                miss_count=miss_count,
                staged_host_locs=stage_sources[0 if expected_tag == 101 else 1][0],
                staged_count=stage_counts[0 if expected_tag == 101 else 1],
                ready_tag=ready_tag,
                expected_tag=expected_tag,
                promotion_src=promotion_src,
                promotion_dst=promotion_dst,
                promotion_count=promotion_count,
                repair_src=repair_src,
                repair_dst=repair_dst,
                repair_count=repair_count,
            )
            copy_cache_planned_mla(
                miss_src=repair_src,
                miss_dst=repair_dst,
                miss_count=repair_count,
                num_real_reqs=num_real_reqs,
                host_cache=host_cache,
                device_buffer=real_buffer,
                item_size_bytes=item_size_bytes,
            )
    urgent_done.record(urgent)
    urgent_done.synchronize()

    # The gated payload and tags are the observable proof that neither
    # producer has run yet.
    assert all(int(tag.item()) == 0 for tag in ready_tags)
    assert all(torch.all(stage == -999) for stage in stage_buffers)
    assert torch.equal(real_buffer[0].cpu(), host_cache[1])
    assert torch.equal(real_buffer[2].cpu(), host_cache[3])
    assert torch.equal(real_buffer[1].cpu(), host_cache[2])
    assert torch.equal(real_buffer[3].cpu(), host_cache[3])

    hold.synchronize()
    for stream in producer_streams:
        stream.synchronize()
    assert all(done.query() for done in producer_done)
    assert all(int(tag.item()) == expected for tag, expected in zip(ready_tags, expected_tags))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
