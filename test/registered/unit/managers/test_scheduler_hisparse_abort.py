import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.hisparse_coordinator import (  # noqa: E402
    HiSparseAct,
    HiSparseCoordinator,
)
from sglang.srt.managers.io_struct import AbortReq, ShutdownReq  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _scheduler_for_abort(staged_req: object, *, duplicate_in_last_batch: bool):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.chunked_req = None
    scheduler.enable_hisparse = True
    scheduler.hisparse_coordinator = MagicMock()
    scheduler.hisparse_coordinator.staging_requests = (staged_req,)
    scheduler.waiting_queue = []
    scheduler.tree_cache = MagicMock()
    scheduler.enable_hicache_storage = False
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.dllm_config = None
    scheduler.grammar_manager = MagicMock()
    scheduler.ps = SimpleNamespace(pp_size=1)
    scheduler.running_batch = SimpleNamespace(reqs=[])
    scheduler.last_batch = (
        SimpleNamespace(reqs=[staged_req]) if duplicate_in_last_batch else None
    )
    scheduler.ipc_channels = MagicMock()
    return scheduler


def _scheduler_for_admission(*, hisparse: bool, staged_reqs: tuple[object, ...]):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.enable_hisparse = hisparse
    scheduler.hisparse_coordinator = MagicMock()
    scheduler.hisparse_coordinator.staging_requests = staged_reqs
    scheduler.req_to_token_pool = MagicMock()
    scheduler.req_to_token_pool.available_size.return_value = 8
    return scheduler


class TestSchedulerHiSparseAbort(unittest.TestCase):
    def test_staging_only_abort_retracts_releases_and_notifies_once(self):
        req = MagicMock()
        req.rid = "staged"
        req.finished.return_value = False
        req.to_finish = None
        scheduler = _scheduler_for_abort(req, duplicate_in_last_batch=False)
        order: list[str] = []
        scheduler.hisparse_coordinator.retract_req.side_effect = lambda _req: (
            order.append("retract")
        )
        scheduler.ipc_channels.send_to_tokenizer.send_output.side_effect = (
            lambda *_args: order.append("notify")
        )

        with patch(
            "sglang.srt.managers.scheduler.release_kv_cache",
            side_effect=lambda *_args, **_kwargs: order.append("release-native"),
        ) as release:
            scheduler.abort_request(AbortReq(rid="staged", exact_match=True))

        self.assertEqual(order, ["retract", "release-native", "notify"])
        scheduler.hisparse_coordinator.retract_req.assert_called_once_with(req)
        release.assert_called_once()

    def test_last_batch_reference_does_not_double_release_staged_request(self):
        req = MagicMock()
        req.rid = "staged"
        req.finished.return_value = False
        req.to_finish = None
        scheduler = _scheduler_for_abort(req, duplicate_in_last_batch=True)

        with patch("sglang.srt.managers.scheduler.release_kv_cache") as release:
            scheduler.abort_request(AbortReq(rid="staged", exact_match=True))

        scheduler.hisparse_coordinator.retract_req.assert_called_once_with(req)
        release.assert_called_once()
        self.assertIsNone(req.to_finish)
        self.assertEqual(
            scheduler.ipc_channels.send_to_tokenizer.send_output.call_count,
            1,
        )

    def test_exact_staging_abort_leaves_prefix_related_request(self):
        target = MagicMock(rid="request")
        sibling = MagicMock(rid="request-child")
        scheduler = _scheduler_for_abort(target, duplicate_in_last_batch=False)
        scheduler.hisparse_coordinator.staging_requests = (target, sibling)

        with patch("sglang.srt.managers.scheduler.release_kv_cache") as release:
            scheduler.abort_request(AbortReq(rid="request", exact_match=True))

        scheduler.hisparse_coordinator.retract_req.assert_called_once_with(target)
        release.assert_called_once()

    def test_queued_exact_abort_leaves_staging_owner_untouched(self):
        staged = MagicMock(rid="active")
        queued = MagicMock(rid="queued")
        queued.mamba_pool_idx = None
        scheduler = _scheduler_for_abort(staged, duplicate_in_last_batch=False)
        scheduler.waiting_queue = [queued]

        with patch("sglang.srt.managers.scheduler.release_kv_cache") as release:
            scheduler.abort_request(AbortReq(rid="queued", exact_match=True))

        self.assertEqual(scheduler.waiting_queue, [])
        scheduler.hisparse_coordinator.retract_req.assert_not_called()
        release.assert_not_called()
        self.assertEqual(
            scheduler.ipc_channels.send_to_tokenizer.send_output.call_count,
            1,
        )


class TestSchedulerHiSparseAdmission(unittest.TestCase):
    def test_staged_request_consumes_b1_admission(self):
        scheduler = _scheduler_for_admission(hisparse=True, staged_reqs=(object(),))

        with patch(
            "sglang.srt.managers.scheduler.get_parallel",
            return_value=SimpleNamespace(pp_max_micro_batch_size=1),
        ):
            self.assertEqual(scheduler.get_num_allocatable_reqs(running_bs=0), 0)

    def test_staged_abort_restores_b1_admission(self):
        req = MagicMock(rid="staged")
        scheduler = _scheduler_for_abort(req, duplicate_in_last_batch=False)
        scheduler.req_to_token_pool = MagicMock()
        scheduler.req_to_token_pool.available_size.return_value = 8
        scheduler.hisparse_coordinator.retract_req.side_effect = lambda _req: setattr(
            scheduler.hisparse_coordinator, "staging_requests", ()
        )

        with (
            patch(
                "sglang.srt.managers.scheduler.get_parallel",
                return_value=SimpleNamespace(pp_max_micro_batch_size=1),
            ),
            patch("sglang.srt.managers.scheduler.release_kv_cache"),
        ):
            self.assertEqual(scheduler.get_num_allocatable_reqs(running_bs=0), 0)
            scheduler.abort_request(AbortReq(rid="staged", exact_match=True))
            self.assertEqual(scheduler.get_num_allocatable_reqs(running_bs=0), 1)

    def test_inactive_native_admission_is_unchanged(self):
        scheduler = _scheduler_for_admission(hisparse=False, staged_reqs=())

        with patch(
            "sglang.srt.managers.scheduler.get_parallel",
            return_value=SimpleNamespace(pp_max_micro_batch_size=1),
        ):
            self.assertEqual(scheduler.get_num_allocatable_reqs(running_bs=0), 1)

    def test_health_idle_check_counts_staging_as_active(self):
        scheduler = _scheduler_for_admission(hisparse=True, staged_reqs=(object(),))
        scheduler.hisparse_coordinator.has_ongoing_staging.return_value = True
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.is_empty.return_value = True
        scheduler.chunked_req = None
        scheduler.dllm_manager = MagicMock()
        scheduler.dllm_manager.any_staging_reqs.return_value = False
        scheduler.last_batch = None
        scheduler.enable_overlap = False
        scheduler._pp_microbatches_drained = MagicMock(return_value=True)
        scheduler.waiting_queue = []
        scheduler._engine_paused = False
        scheduler.disaggregation_mode = DisaggregationMode.NULL

        self.assertFalse(scheduler.is_fully_idle(for_health_check=True))

    def test_staging_only_owner_blocks_destructive_cache_flush(self):
        scheduler = _scheduler_for_admission(hisparse=True, staged_reqs=(object(),))
        scheduler.hisparse_coordinator.has_ongoing_staging.return_value = True
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.is_empty.return_value = True
        scheduler.chunked_req = None
        scheduler.dllm_manager = MagicMock()
        scheduler.dllm_manager.any_staging_reqs.return_value = False
        scheduler.last_batch = None
        scheduler.enable_overlap = False
        scheduler._pp_microbatches_drained = MagicMock(return_value=True)
        scheduler.waiting_queue = []
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.enable_hierarchical_cache = False
        scheduler.tree_cache = MagicMock()
        scheduler.token_to_kv_pool_allocator = MagicMock()
        scheduler.grammar_manager = MagicMock()
        scheduler.metrics_reporter = MagicMock()
        scheduler.draft_worker = None

        self.assertFalse(scheduler.flush_cache(empty_cache=False))
        scheduler.tree_cache.reset.assert_not_called()
        scheduler.req_to_token_pool.clear.assert_not_called()
        scheduler.token_to_kv_pool_allocator.clear.assert_not_called()


class TestSchedulerHiSparseShutdown(unittest.TestCase):
    def test_graceful_teardown_drains_hisparse_before_stock_host_resources(self):
        scheduler = Scheduler.__new__(Scheduler)
        order: list[str] = []
        scheduler.gracefully_exit = False
        scheduler.hisparse_coordinator = MagicMock()
        scheduler.hisparse_coordinator.destroy.side_effect = lambda: order.append(
            "drain-hisparse"
        )
        scheduler.tree_cache = MagicMock()
        scheduler.tree_cache.release_host_resources.side_effect = lambda: order.append(
            "release-tree"
        )
        scheduler.decode_offload_manager = MagicMock()
        scheduler.decode_offload_manager.release_host_resources.side_effect = lambda: (
            order.append("release-offload")
        )

        scheduler.handle_shutdown(ShutdownReq())
        scheduler.release_host_resources()

        self.assertTrue(scheduler.gracefully_exit)
        self.assertEqual(
            order,
            ["drain-hisparse", "release-tree", "release-offload"],
        )


class TestHiSparseStagingRetirement(unittest.TestCase):
    def test_transfer_and_split_work_retire_before_storage_release(self):
        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        req = SimpleNamespace(
            rid="staged",
            req_pool_idx=0,
            extend_range=SimpleNamespace(end=2),
            hisparse_staging=True,
        )
        coordinator.ack_staging_queue = [HiSparseAct(None, None, req)]
        coordinator._overlap_enabled = True
        order: list[str] = []
        coordinator._drain_split_materialization = MagicMock(
            side_effect=lambda: order.append("drain-split")
        )
        coordinator.write_staging_stream = MagicMock()
        coordinator.write_staging_stream.synchronize.side_effect = lambda: order.append(
            "drain-prefill"
        )
        coordinator.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.tensor([[11, 12]], dtype=torch.int64)
        )
        coordinator.token_to_kv_pool_allocator = SimpleNamespace(
            free_hisparse=lambda _locs: order.append("free-device")
        )
        coordinator.req_to_host_pool = torch.tensor([[21, 22]], dtype=torch.int64)
        coordinator.req_to_host_pool_allocated_len = torch.tensor(
            [2], dtype=torch.int64
        )
        coordinator.mem_pool_host = SimpleNamespace(
            allocated_host_indices=lambda *_args: torch.tensor(
                [21, 22], dtype=torch.int64
            ),
            free=lambda _locs: order.append("free-host"),
        )
        coordinator._skip_first_backup = [True]

        coordinator.abort_staging_request(req)

        self.assertEqual(
            order,
            ["drain-split", "drain-prefill", "free-device", "free-host"],
        )
        self.assertEqual(coordinator.ack_staging_queue, [])
        self.assertFalse(req.hisparse_staging)
        self.assertEqual(
            int(coordinator.req_to_host_pool_allocated_len[req.req_pool_idx]), 0
        )

    def test_destroy_drains_all_owned_streams_before_host_unregister(self):
        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        order: list[str] = []
        coordinator._split_materialization_enabled = True
        coordinator._drain_split_materialization = MagicMock(
            side_effect=lambda: order.append("drain-split")
        )
        coordinator.write_staging_stream = MagicMock()
        coordinator.write_staging_stream.synchronize.side_effect = lambda: order.append(
            "drain-prefill"
        )
        coordinator.decode_backup_stream = MagicMock()
        coordinator.decode_backup_stream.synchronize.side_effect = lambda: order.append(
            "drain-backup"
        )
        coordinator.enable_prefetch = True
        coordinator.prefetch_stream = MagicMock()
        coordinator.prefetch_stream.synchronize.side_effect = lambda: order.append(
            "drain-prefetch"
        )
        coordinator.mem_pool_host = MagicMock()
        coordinator.mem_pool_host.destroy.side_effect = lambda: order.append(
            "unregister-host"
        )

        coordinator.destroy()

        self.assertEqual(
            order,
            [
                "drain-split",
                "drain-prefill",
                "drain-backup",
                "drain-prefetch",
                "unregister-host",
            ],
        )


if __name__ == "__main__":
    unittest.main()
