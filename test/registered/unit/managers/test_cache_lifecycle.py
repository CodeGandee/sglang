"""Scheduler-facing tests for generic request-cache lifecycle callbacks."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.benchmark.one_batch import _TorchBenchRunner
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.cache_lifecycle import (
    CacheAllocation,
    CacheBatchMutation,
    CacheBatchMutationKind,
    CacheTerminalEvent,
    CacheTerminalReason,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FinishReason:
    def __init__(self, kind: str):
        self.kind = kind

    def to_json(self):
        return {"type": self.kind}


class _RecordingCallbacks:
    def __init__(self, *, fail_publication: bool = False):
        self.events = []
        self.fail_publication = fail_publication

    def on_request_admitted(self, allocation: CacheAllocation) -> None:
        self.events.append(("admitted", allocation))

    def on_request_published(self, allocation: CacheAllocation) -> None:
        self.events.append(("published", allocation))
        if self.fail_publication:
            raise RuntimeError("injected publication failure")

    def on_batch_mutated(self, mutation: CacheBatchMutation) -> None:
        self.events.append(("batch", mutation))

    def on_request_terminal(self, event: CacheTerminalEvent) -> None:
        self.events.append(("terminal", event))


def _make_req(rid: str):
    return SimpleNamespace(
        rid=rid,
        req_pool_idx=None,
        inflight_middle_chunks=0,
        kv_committed_len=0,
        finished_reason=None,
        is_retracted=False,
    )


def _make_pool(callbacks: _RecordingCallbacks, *, size: int = 2):
    pool = ReqToTokenPool(
        size=size,
        max_context_len=8,
        device="cpu",
        enable_memory_saver=False,
    )
    pool.bind_cache_lifecycle(callbacks)
    return pool


class TestCacheLifecycle(unittest.TestCase):
    def test_one_batch_cleanup_terminates_allocations_before_pool_reset(self):
        events = []
        request = SimpleNamespace(req_pool_idx=1)

        class _RequestPool:
            def free(self, req, *, terminal_reason, terminal_detail):
                events.append(("free", terminal_reason, terminal_detail))
                req.req_pool_idx = None

            def clear(self):
                self.assert_no_active_request()
                events.append(("request-pool-clear",))

            def assert_no_active_request(self):
                if request.req_pool_idx is not None:
                    raise RuntimeError("request allocation is still active")

        class _TokenAllocator:
            def clear(self):
                events.append(("token-pool-clear",))

        runner = _TorchBenchRunner(
            SimpleNamespace(
                req_to_token_pool=_RequestPool(),
                token_to_kv_pool_allocator=_TokenAllocator(),
            )
        )

        runner.cleanup(SimpleNamespace(reqs=[request]))
        runner.clear()

        self.assertEqual(
            events,
            [
                (
                    "free",
                    CacheTerminalReason.COMPLETION,
                    "one-batch benchmark cleanup",
                ),
                ("request-pool-clear",),
                ("token-pool-clear",),
            ],
        )

    def test_completion_precedes_batch_filter(self):
        callbacks = _RecordingCallbacks()
        pool = _make_pool(callbacks)
        req = _make_req("completion")

        pool.alloc([req])
        allocation = pool.cache_allocation(req)
        pool.publish_cache_allocations([req])
        req.finished_reason = _FinishReason("length")
        pool.free(req)

        batch = ScheduleBatch(reqs=[req], req_to_token_pool=pool)
        batch.filter_batch(keep_indices=[])

        self.assertEqual(
            [event[0] for event in callbacks.events],
            ["admitted", "published", "terminal", "batch"],
        )
        terminal = callbacks.events[2][1]
        self.assertEqual(terminal.allocation, allocation)
        self.assertEqual(terminal.reason, CacheTerminalReason.COMPLETION)
        mutation = callbacks.events[3][1]
        self.assertEqual(mutation.kind, CacheBatchMutationKind.FILTER)
        self.assertEqual(mutation.allocations, ())

    def test_abort_is_inferred_from_finish_metadata(self):
        callbacks = _RecordingCallbacks()
        pool = _make_pool(callbacks)
        req = _make_req("abort")

        pool.alloc([req])
        pool.publish_cache_allocations([req])
        req.finished_reason = _FinishReason("abort")
        pool.free(req)

        terminal = callbacks.events[-1][1]
        self.assertEqual(terminal.reason, CacheTerminalReason.ABORT)

    def test_retraction_reuses_slot_with_a_new_generation(self):
        callbacks = _RecordingCallbacks()
        pool = _make_pool(callbacks, size=1)
        first = _make_req("first")

        pool.alloc([first])
        first_allocation = pool.cache_allocation(first)
        pool.publish_cache_allocations([first])
        pool.free(first, terminal_reason=CacheTerminalReason.RETRACTION)

        second = _make_req("second")
        pool.alloc([second])
        second_allocation = pool.cache_allocation(second)

        self.assertEqual(
            second_allocation.request_pool_index,
            first_allocation.request_pool_index,
        )
        self.assertEqual(second_allocation.generation, first_allocation.generation + 1)
        terminal = [event for kind, event in callbacks.events if kind == "terminal"]
        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0].reason, CacheTerminalReason.RETRACTION)

    def test_idle_pool_clear_preserves_monotonic_slot_generation(self):
        callbacks = _RecordingCallbacks()
        pool = _make_pool(callbacks, size=1)
        first = _make_req("before-clear")

        pool.alloc([first])
        first_allocation = pool.cache_allocation(first)
        pool.publish_cache_allocations([first])
        pool.free(first, terminal_reason=CacheTerminalReason.COMPLETION)
        pool.clear()

        second = _make_req("after-clear")
        pool.alloc([second])
        second_allocation = pool.cache_allocation(second)

        self.assertEqual(
            second_allocation.request_pool_index,
            first_allocation.request_pool_index,
        )
        self.assertEqual(second_allocation.generation, first_allocation.generation + 1)

    def test_publication_failure_notifies_terminal_before_raising(self):
        callbacks = _RecordingCallbacks(fail_publication=True)
        pool = _make_pool(callbacks)
        req = _make_req("failure")

        pool.alloc([req])
        with self.assertRaisesRegex(RuntimeError, "injected publication failure"):
            pool.publish_cache_allocations([req])

        self.assertEqual(
            [event[0] for event in callbacks.events],
            ["admitted", "published", "terminal"],
        )
        terminal = callbacks.events[-1][1]
        self.assertEqual(terminal.reason, CacheTerminalReason.FAILURE)
        self.assertIn("injected publication failure", terminal.detail)
        pool.free(req)

    def test_duplicate_terminal_notifications_are_harmless(self):
        callbacks = _RecordingCallbacks()
        pool = _make_pool(callbacks)
        req = _make_req("duplicate")

        pool.alloc([req])
        pool.publish_cache_allocations([req])
        first = pool.notify_cache_terminal(req, CacheTerminalReason.ABORT)
        duplicate = pool.notify_cache_terminal(req, CacheTerminalReason.FAILURE)
        pool.free(req, terminal_reason=CacheTerminalReason.COMPLETION)

        self.assertTrue(first)
        self.assertFalse(duplicate)
        terminal = [event for kind, event in callbacks.events if kind == "terminal"]
        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0].reason, CacheTerminalReason.ABORT)


if __name__ == "__main__":
    unittest.main()
