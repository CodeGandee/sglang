"""CPU owning-boundary regressions for live backing and allocator publication."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.kv_vmm_backing import (
    KvVmmArena,
    KvVmmBufferOwner,
    _BufferSpec,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Desc:
    def __init__(self, name, row_bytes):
        self.name = name
        self.row_bytes = row_bytes

    def final_span_bytes(self, tokens, page_size):
        return (tokens + page_size) * self.row_bytes


class _Arena:
    granularity = 16

    def __init__(self):
        self.commits = {}
        self.fail_offset = None
        self.closed = False

    def commit_range(self, offset, size):
        if offset == self.fail_offset:
            raise RuntimeError("injected second-buffer mapping refusal")
        self.commits[offset] = size

    @property
    def backed_bytes(self):
        return sum(self.commits.values())

    def close(self):
        self.closed = True


def _owner():
    # Use the actual owner growth implementation with deterministic mappings;
    # only construction/driver calls are replaced in this CPU regression.
    owner = object.__new__(KvVmmBufferOwner)
    owner.page_size = 4
    owner._reserved_num_tokens = 16
    owner._usable_num_tokens = 0
    owner._final_num_tokens = None
    owner._arena = _Arena()
    owner._specs = [
        _BufferSpec(_Desc("records", 4), 0, 80, 80),
        _BufferSpec(_Desc("scales", 2), 128, 40, 48),
    ]
    owner.tensors = []
    owner.grow_prefix(4)
    return owner


class TestLiveAllocatorGrowth(unittest.TestCase):
    def test_preparation_does_not_publish_and_commit_allocates_nothing(self):
        allocator = PagedTokenToKVPoolAllocator(
            16, 4, torch.float32, "cpu", None, False
        )
        live = allocator.alloc(4)
        prepared = allocator.prepare_grow(32)
        self.assertEqual(allocator.size, 16)
        self.assertEqual(allocator.available_size(), 12)
        with (
            patch.object(
                torch, "arange", side_effect=AssertionError("late allocation")
            ),
            patch.object(torch, "cat", side_effect=AssertionError("late allocation")),
        ):
            allocator.commit_grow(prepared)
        self.assertEqual(allocator.size, 32)
        self.assertEqual(allocator.num_pages, 8)
        self.assertEqual(live.tolist(), [4, 5, 6, 7])
        with self.assertRaises(ValueError):
            allocator.commit_grow(prepared)

    def test_intervening_allocation_refuses_stale_prepared_page_list(self):
        allocator = PagedTokenToKVPoolAllocator(
            16, 4, torch.float32, "cpu", None, False
        )
        prepared = allocator.prepare_grow(32)
        allocator.alloc(4)
        with self.assertRaises(ValueError):
            allocator.commit_grow(prepared)
        self.assertEqual(allocator.size, 16)

    def test_paged_growth_preserves_active_and_deferred_pages(self):
        for sort in (False, True):
            with self.subTest(sort=sort):
                allocator = PagedTokenToKVPoolAllocator(
                    16, 4, torch.float32, "cpu", None, sort
                )
                a = allocator.alloc(4)
                b = allocator.alloc(4)
                original_a = a.clone()
                allocator.free_group_begin()
                allocator.free(b)
                allocator.grow(32)
                allocator.grow(32)
                self.assertFalse(allocator.is_not_in_free_group)
                self.assertEqual(allocator.num_pages, 8)
                allocator.free_group_end()
                allocator.merge_and_sort_free()
                pages = allocator.free_pages.tolist()
                self.assertEqual(len(pages), len(set(pages)))
                self.assertNotIn(1, pages)
                self.assertEqual(set(pages), set(range(2, 9)))
                self.assertTrue(torch.equal(a, original_a))
                allocator.free(a)
                allocator.merge_and_sort_free()
                self.assertEqual(allocator.available_size(), 32)
                self.assertEqual(set(allocator.free_pages.tolist()), set(range(1, 9)))

    def test_token_growth_and_invalid_capacity_leave_live_slots(self):
        allocator = TokenToKVPoolAllocator(4, torch.float32, "cpu", None, True)
        live = allocator.alloc(2)
        allocator.grow(8)
        with self.assertRaises(ValueError):
            allocator.grow(4)
        self.assertEqual(live.tolist(), [1, 2])
        self.assertEqual(allocator.free_pages.tolist(), [3, 4, 5, 6, 7, 8])

    def test_rounding_refusal_is_not_destructive_resize(self):
        allocator = PagedTokenToKVPoolAllocator(
            16, 4, torch.float32, "cpu", None, False
        )
        live = allocator.alloc(4)
        before = allocator.free_pages.clone()
        with self.assertRaises(ValueError):
            allocator.grow(17)
        self.assertEqual(allocator.size, 16)
        self.assertTrue(torch.equal(before, allocator.free_pages))
        self.assertEqual(live.tolist(), [4, 5, 6, 7])


class TestLiveVmmGrowth(unittest.TestCase):
    def test_teardown_selects_owned_device_and_closes_once(self):
        calls = []

        class Context:
            def __enter__(self):
                calls.append("enter-device-3")

            def __exit__(self, *args):
                calls.append("exit-device-3")

        arena = object.__new__(KvVmmArena)
        arena._closed = False
        arena.device_id = 3
        arena._allocation = SimpleNamespace(close=lambda: calls.append("driver-close"))
        with (
            patch.object(torch.cuda, "device", return_value=Context()) as device,
            patch.object(
                torch.cuda,
                "synchronize",
                side_effect=lambda ordinal: calls.append(f"sync-{ordinal}"),
            ),
        ):
            arena.close()
            arena.close()
            device.assert_called_once_with(3)
        self.assertEqual(
            calls, ["enter-device-3", "sync-3", "driver-close", "exit-device-3"]
        )

    def test_construction_failure_closes_partial_mappings(self):
        arena = _Arena()
        arena.base = 0
        arena.reserved = 4096
        arena.pool = object()
        arena.fail_offset = 128
        descriptors = []
        tensors = []
        for offset in (0, 128):
            desc = _Desc(str(offset), 4)
            desc.shape = (20,)
            desc.prefix_span_bytes = lambda count, page: count * 4
            desc.reserved_span_bytes = lambda itemsize: 20 * itemsize
            descriptors.append(desc)
            tensors.append(SimpleNamespace(shape=(20,), data_ptr=lambda o=offset: o))
        with (
            patch("sglang.srt.mem_cache.kv_vmm_backing.KvVmmArena", return_value=arena),
            patch(
                "sglang.srt.mem_cache.kv_vmm_backing.get_device_granularity",
                return_value=16,
            ),
            patch.object(torch.cuda, "device", return_value=nullcontext()),
            patch.object(torch.cuda, "use_mem_pool", return_value=nullcontext()),
            patch.object(torch, "empty", side_effect=tensors),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                KvVmmBufferOwner(
                    device="cuda:0",
                    device_id=0,
                    store_dtype=torch.float32,
                    page_size=4,
                    reserved_num_tokens=16,
                    buffer_descs=descriptors,
                )
        self.assertTrue(arena.closed)

    def test_partial_commit_is_accounted_but_not_published(self):
        owner = _owner()
        arena = owner._arena
        before = owner.backed_bytes
        arena.fail_offset = 128
        with self.assertRaisesRegex(RuntimeError, "injected"):
            owner.grow_prefix(12)
        self.assertEqual(owner.usable_num_tokens, 4)
        states = owner.buffer_states
        self.assertEqual(states[0].committed_bytes, 64)
        self.assertEqual(states[0].usable_bytes, 32)
        self.assertEqual(states[1].committed_bytes, 16)
        self.assertEqual(owner.backed_bytes, before + 32)
        arena.fail_offset = None
        owner.grow_prefix(12)
        self.assertEqual(owner.usable_num_tokens, 12)
        self.assertEqual(owner.backed_bytes, 96)
        self.assertEqual(owner.buffer_states[0].committed_bytes, 64)

    def test_span_vector_and_ceiling_refuse_before_commit(self):
        owner = _owner()
        before = owner.backed_bytes
        for count in (3, 20, 0):
            with self.assertRaises(ValueError):
                owner.grow_prefix(count)
        with self.assertRaises(ValueError):
            owner._back_spans([40])
        with self.assertRaises(ValueError):
            owner._back_spans([40, 100])
        self.assertEqual(owner.backed_bytes, before)
        self.assertEqual(owner.usable_num_tokens, 4)

    def test_close_releases_partial_owner_and_is_idempotent(self):
        owner = _owner()
        arena = owner._arena
        arena.fail_offset = 128
        with self.assertRaises(RuntimeError):
            owner.grow_prefix(12)
        owner.close()
        owner.close()
        self.assertTrue(arena.closed)
        self.assertEqual(owner.backed_bytes, 0)
        self.assertEqual(owner.usable_num_tokens, 0)
        self.assertEqual(owner.buffer_states, ())


class TestHiSparseGrowthPublication(unittest.TestCase):
    def test_logical_indexer_and_mapping_ceiling_stays_fixed(self):
        from sglang.srt.mem_cache.allocator.hisparse import (
            HiSparseTokenToKVPoolAllocator,
        )

        pool = SimpleNamespace(
            size=256,
            backed_device_tokens=128,
            register_mapping=lambda mapping: None,
        )
        allocator = HiSparseTokenToKVPoolAllocator(
            256, 64, torch.float32, "cpu", pool, False, host_to_device_ratio=2
        )
        self.assertEqual(allocator.size_full, 512)
        self.assertEqual(allocator.logical_attn_allocator.size, 512)
        self.assertEqual(allocator.hisparse_attn_allocator.size, 128)
        mapping = allocator.full_to_hisparse_device_index_mapping
        pointer = mapping.data_ptr()
        mapping[3] = 70
        active = allocator.hisparse_attn_allocator.alloc(128)
        with self.assertRaises(ValueError):
            allocator.publish_device_growth(192)
        self.assertEqual(allocator.hisparse_attn_allocator.size, 128)
        pool.backed_device_tokens = 192
        allocator.publish_device_growth(192)
        new = allocator.hisparse_attn_allocator.alloc(64)
        self.assertEqual(new.tolist(), list(range(192, 256)))
        self.assertEqual(active.tolist(), list(range(64, 192)))
        self.assertEqual(mapping.data_ptr(), pointer)
        self.assertEqual(mapping[3].item(), 70)
        self.assertEqual(allocator.logical_attn_allocator.size, 512)

    def test_hisparse_descriptor_uses_actual_packed_geometry(self):
        from sglang.srt.mem_cache.hisparse_memory_pool import (
            HiSparseDSATokenToKVPool,
        )

        captured = {}

        class Owner:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.usable_num_tokens = 0
                self.tensors = [
                    torch.zeros(desc.shape, dtype=kwargs["store_dtype"])
                    for desc in kwargs["buffer_descs"]
                ]

            def grow_prefix(self, tokens):
                self.usable_num_tokens = tokens

        pool = object.__new__(HiSparseDSATokenToKVPool)
        pool.size = 256
        pool.page_size = 64
        pool.kv_cache_dim = 713  # packed scales/rope bytes; no fixed GLM width
        pool.store_dtype = torch.uint8
        pool.layer_num = 2
        pool.start_layer = 17
        pool.device = "cuda:0"
        pool.custom_mem_pool = None
        pool._live_growth_initial_tokens = 128
        pool._live_growth_owner = None
        with patch("sglang.srt.mem_cache.hisparse_memory_pool.KvVmmBufferOwner", Owner):
            pool._create_buffers()
        self.assertEqual(pool.backed_device_tokens, 128)
        self.assertEqual(pool.size, 256)
        descs = captured["buffer_descs"]
        self.assertEqual(descs[0].name, "hisparse-layer-17")
        self.assertEqual(descs[0].shape, (320, 1, 713))
        self.assertEqual(descs[1].row_bytes, 713)
        self.assertEqual(descs[0].final_span_bytes(128, 64), 192 * 713)
        pool.prepare_device_growth(192)
        self.assertEqual(pool.backed_device_tokens, 192)


if __name__ == "__main__":
    unittest.main()
