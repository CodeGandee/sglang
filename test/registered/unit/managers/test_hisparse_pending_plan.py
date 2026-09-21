from __future__ import annotations

import unittest
from contextlib import nullcontext
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.managers import hisparse_coordinator as hisparse_module
from sglang.srt.managers.hisparse_coordinator import (
    HiSparseCoordinator,
    _HiSparseRequestIdentity,
)


class _FakeStream:
    def __init__(self) -> None:
        self.waited: list[object] = []
        self.operations: list[str] = []
        self.synchronize_count = 0
        self.fail_synchronize = False

    def wait_stream(self, stream: object) -> None:
        self.waited.append(stream)
        self.operations.append("wait-stream")

    def synchronize(self) -> None:
        self.operations.append("synchronize")
        self.synchronize_count += 1
        if self.fail_synchronize:
            raise RuntimeError("injected fence failure")


class _FakeEvent:
    def __init__(self) -> None:
        self.recorded: list[object] = []
        self.waited: list[object] = []

    def record(self, stream: object) -> None:
        self.recorded.append(stream)

    def wait(self, stream: object) -> None:
        self.waited.append(stream)


class _FakeDeviceModule:
    def __init__(self) -> None:
        self.compute_stream = _FakeStream()

    def current_stream(self) -> _FakeStream:
        return self.compute_stream

    @staticmethod
    def stream(_stream: object) -> nullcontext[None]:
        return nullcontext()


def _coordinator() -> HiSparseCoordinator:
    coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
    coordinator.top_k = 2
    coordinator.device_buffer_size = 4
    coordinator.swap_in_block_size = 8
    coordinator.num_real_reqs = torch.tensor([1], dtype=torch.int32)
    coordinator.top_k_device_locs_buffer = torch.full((1, 2), -1, dtype=torch.int32)
    coordinator.req_device_buffer_tokens = torch.stack(
        [torch.full((1, 5), layer, dtype=torch.int32) for layer in range(10)]
    )
    coordinator.req_to_host_pool = torch.arange(16, dtype=torch.int64).view(1, -1)
    coordinator.req_device_buffer_token_locs = torch.zeros(
        (10, 1, 5), dtype=torch.int32
    )
    coordinator.lru_slots = torch.zeros((10, 1, 4), dtype=torch.int16)
    coordinator._miss_src = torch.zeros((1, 2), dtype=torch.int64)
    coordinator._miss_dst = torch.zeros((1, 2), dtype=torch.int32)
    coordinator._miss_count = torch.zeros((1,), dtype=torch.int32)
    coordinator._is_shared_index_layer = [
        False,
        False,
        False,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
    ]
    coordinator._prefetch_groups = {2: [3, 4, 5], 6: [7, 8, 9]}
    coordinator._prefetch_slot = [0, 0, 0, 0, 1, 2, 0, 0, 1, 2]
    coordinator._prefetch_events = [_FakeEvent(), _FakeEvent(), _FakeEvent()]
    coordinator.prefetch_stream = _FakeStream()
    coordinator._split_worker_reusable = True
    coordinator._split_generation = 1
    identity = _HiSparseRequestIdentity(7, 1, 101)
    coordinator._split_requests = {7: identity}
    coordinator._split_step_request = identity
    coordinator._split_next_request = identity
    coordinator._split_aborted_generation = None
    coordinator._split_step = 0
    coordinator._split_anchor_index = 0
    coordinator._split_step_request_ptr = None
    coordinator._split_anchor_layers = (0, 1, 2, 6)
    coordinator._split_pending = None
    coordinator._split_followers_seen = 0
    coordinator._split_materialization_enabled = True
    coordinator._has_pending_backup = False
    coordinator.decode_producer_stream = None
    return coordinator


class TestHiSparsePendingPlan(unittest.TestCase):
    def test_every_anchor_materializes_and_one_borrowed_plan_is_reused(self) -> None:
        coordinator = _coordinator()
        copies: list[int] = []
        plans: list[int] = []
        operations: list[str] = []

        def copy_only(_self: object, _num_reqs: int, layer_id: int) -> None:
            copies.append(layer_id)
            operations.append(f"copy:{layer_id}")

        coordinator._run_copy_only_kernel = MethodType(copy_only, coordinator)
        coordinator.wait_for_pending_backup = MethodType(
            lambda _self: operations.append("wait-backup"), coordinator
        )

        def plan_only(**kwargs: torch.Tensor | int) -> None:
            layer_id = int(kwargs["device_buffer_tokens"][0, 0])
            plans.append(layer_id)
            operations.append(f"plan:{layer_id}")
            kwargs["top_k_device_locs"].copy_(
                torch.tensor([[layer_id, layer_id + 20]], dtype=torch.int32)
            )
            kwargs["miss_src"].copy_(torch.tensor([[31, 37]], dtype=torch.int64))
            kwargs["miss_dst"].copy_(torch.tensor([[3, 4]], dtype=torch.int32))
            kwargs["miss_count"].fill_(2)

        request_indices = torch.tensor([7], dtype=torch.int64)
        seq_lens = torch.tensor([5], dtype=torch.int32)
        selected = torch.tensor([[1, 2]], dtype=torch.int32)
        fake_device = _FakeDeviceModule()
        tables: list[torch.Tensor] = []
        with (
            patch.object(hisparse_module, "device_module", fake_device),
            patch.object(
                hisparse_module,
                "plan_cache_to_device_buffer_mla",
                side_effect=plan_only,
            ),
        ):
            for layer_id in range(10):
                tables.append(
                    coordinator.swap_in_selected_pages(
                        request_indices, seq_lens, selected, layer_id
                    )
                )

            previous = coordinator._split_pending
            self.assertIsNotNone(previous)
            self.assertEqual(previous.step, 1)
            self.assertEqual(previous.group_layers, (6, 7, 8, 9))
            self.assertEqual(previous.request.generation, 1)
            self.assertEqual(previous.representation, "mla-bf16-width-576")
            next_request = SimpleNamespace(req_pool_idx=5)
            next_identity = _HiSparseRequestIdentity(5, 2, id(next_request))
            coordinator._split_requests[5] = next_identity
            coordinator._bind_split_step_request(torch.tensor([5], dtype=torch.int64))
            coordinator.swap_in_selected_pages(
                torch.tensor([5], dtype=torch.int64),
                seq_lens,
                selected,
                layer_id=0,
            )

        self.assertEqual(plans, [0, 1, 2, 6, 0])
        self.assertEqual(copies, list(range(10)) + [0])
        for anchor in (0, 1, 2, 6):
            plan_index = operations.index(f"plan:{anchor}")
            self.assertEqual(operations[plan_index + 1], "wait-backup")
            self.assertEqual(operations[plan_index + 2], f"copy:{anchor}")
        self.assertTrue(
            all(
                table.data_ptr() == coordinator.top_k_device_locs_buffer[:1].data_ptr()
                for table in tables
            )
        )
        self.assertEqual(coordinator._split_pending.step, 2)
        self.assertEqual(coordinator._split_pending.request.generation, 2)
        with self.assertRaisesRegex(RuntimeError, "stale HiSparse pending-plan"):
            coordinator._validate_split_plan(previous)

    def test_completed_prepared_batch_reenters_only_with_same_identity(self) -> None:
        def plan_only(**kwargs: torch.Tensor | int) -> None:
            kwargs["top_k_device_locs"].fill_(3)
            kwargs["miss_count"].zero_()

        def prepare() -> tuple[HiSparseCoordinator, torch.Tensor, SimpleNamespace]:
            coordinator = _coordinator()
            request = SimpleNamespace(req_pool_idx=7)
            identity = _HiSparseRequestIdentity(7, 1, id(request))
            coordinator._split_requests = {7: identity}
            coordinator._split_step_request = identity
            coordinator._split_next_request = identity
            coordinator._run_copy_only_kernel = MethodType(
                lambda _self, _num_reqs, _layer_id: None, coordinator
            )
            request_indices = torch.tensor([7], dtype=torch.int64)
            seq_lens = torch.tensor([5], dtype=torch.int32)
            selected = torch.tensor([[1, 2]], dtype=torch.int32)
            for layer_id in range(10):
                coordinator.swap_in_selected_pages(
                    request_indices, seq_lens, selected, layer_id
                )
            return coordinator, request_indices, request

        fake_device = _FakeDeviceModule()
        with (
            patch.object(hisparse_module, "device_module", fake_device),
            patch.object(
                hisparse_module,
                "plan_cache_to_device_buffer_mla",
                side_effect=plan_only,
            ),
        ):
            coordinator, request_indices, request = prepare()
            seq_lens = torch.tensor([5], dtype=torch.int32)
            selected = torch.tensor([[1, 2]], dtype=torch.int32)
            for _ in range(3):
                for layer_id in range(10):
                    coordinator.swap_in_selected_pages(
                        request_indices, seq_lens, selected, layer_id
                    )
            self.assertEqual(coordinator._split_pending.step, 4)
            self.assertEqual(coordinator._split_pending.request.generation, 1)
            coordinator._prepare_split_request_release(request)
            coordinator._finish_split_request_release(request)
            self.assertEqual(coordinator._split_requests, {})
            self.assertIsNone(coordinator._split_pending)

            changed, original_indices, _request = prepare()
            changed_indices = original_indices.clone()
            self.assertNotEqual(changed_indices.data_ptr(), original_indices.data_ptr())
            with self.assertRaisesRegex(
                RuntimeError, "completed-step reentry changed request identity"
            ):
                changed.swap_in_selected_pages(
                    changed_indices,
                    torch.tensor([5], dtype=torch.int32),
                    torch.tensor([[1, 2]], dtype=torch.int32),
                    layer_id=0,
                )
            self.assertEqual(changed._split_aborted_generation, 1)

            rebound, original_indices, request = prepare()
            rebound_indices = original_indices.clone()
            rebound.bind_split_materialization_request(request)
            for layer_id in range(10):
                rebound.swap_in_selected_pages(
                    rebound_indices,
                    torch.tensor([5], dtype=torch.int32),
                    torch.tensor([[1, 2]], dtype=torch.int32),
                    layer_id,
                )
            self.assertEqual(rebound._split_pending.step, 2)
            self.assertEqual(
                rebound._split_pending.request_indices_ptr,
                rebound_indices.data_ptr(),
            )

            stale_request = SimpleNamespace(req_pool_idx=7)
            with self.assertRaisesRegex(RuntimeError, "stale request generation"):
                rebound.bind_split_materialization_request(stale_request)

            partial = _coordinator()
            partial_request = SimpleNamespace(req_pool_idx=7)
            partial_identity = _HiSparseRequestIdentity(7, 1, id(partial_request))
            partial._split_requests = {7: partial_identity}
            partial._split_step_request = partial_identity
            partial._split_next_request = partial_identity
            partial._run_copy_only_kernel = MethodType(
                lambda _self, _num_reqs, _layer_id: None, partial
            )
            seq_lens = torch.tensor([5], dtype=torch.int32)
            selected = torch.tensor([[1, 2]], dtype=torch.int32)
            for layer_id in (0, 1, 2):
                partial.swap_in_selected_pages(
                    request_indices, seq_lens, selected, layer_id
                )
            with self.assertRaisesRegex(RuntimeError, "partial step"):
                partial.bind_split_materialization_request(partial_request)
            with self.assertRaisesRegex(RuntimeError, "before every follower"):
                partial.swap_in_selected_pages(
                    request_indices, seq_lens, selected, layer_id=6
                )

    def test_abort_drains_safe_work_and_failed_fence_quarantines_worker(self) -> None:
        coordinator = _coordinator()
        coordinator._split_pending = SimpleNamespace()
        fake_device = _FakeDeviceModule()
        producer_stream = _FakeStream()
        coordinator.decode_producer_stream = producer_stream
        with patch.object(hisparse_module, "device_module", fake_device):
            coordinator.abort_split_materialization(safe_to_reuse=True)

        self.assertIsNone(coordinator._split_pending)
        self.assertEqual(coordinator._split_aborted_generation, 1)
        self.assertEqual(coordinator.prefetch_stream.synchronize_count, 1)
        self.assertEqual(fake_device.compute_stream.synchronize_count, 1)
        self.assertEqual(fake_device.compute_stream.waited, [producer_stream])
        self.assertEqual(
            fake_device.compute_stream.operations, ["wait-stream", "synchronize"]
        )

        coordinator._split_aborted_generation = None
        coordinator._split_pending = SimpleNamespace()
        coordinator.prefetch_stream.fail_synchronize = True
        with (
            patch.object(hisparse_module, "device_module", fake_device),
            self.assertRaisesRegex(RuntimeError, "injected fence failure"),
        ):
            coordinator.abort_split_materialization(safe_to_reuse=True)
        self.assertFalse(coordinator.split_worker_reusable)
        with self.assertRaisesRegex(RuntimeError, "unsafe; retire it"):
            coordinator._assert_split_worker_reusable()
        req = SimpleNamespace(req_pool_idx=7)
        unsafe_identity = _HiSparseRequestIdentity(7, 1, id(req))
        coordinator._split_requests = {7: unsafe_identity}
        coordinator._split_step_request = unsafe_identity
        with self.assertRaisesRegex(RuntimeError, "unsafe; retire it"):
            coordinator._prepare_split_request_release(req)

    def test_allocated_requests_keep_independent_slot_generations(self) -> None:
        coordinator = _coordinator()
        coordinator._split_generation = 0
        coordinator._split_requests = {}
        coordinator._split_step_request = None
        coordinator._split_next_request = None
        first = SimpleNamespace(req_pool_idx=2)
        second = SimpleNamespace(req_pool_idx=5)

        coordinator._assert_can_activate_split_request(first)
        coordinator._activate_split_request(first)
        coordinator._assert_can_activate_split_request(second)
        coordinator._activate_split_request(second)
        coordinator._bind_split_step_request(torch.tensor([5], dtype=torch.int64))

        self.assertEqual(set(coordinator._split_requests), {2, 5})
        self.assertEqual(coordinator._split_requests[2].generation, 1)
        self.assertEqual(coordinator._split_requests[5].generation, 2)
        self.assertEqual(coordinator._split_next_request.slot, 5)

    def test_release_drains_before_free_and_reuses_generation_once(self) -> None:
        coordinator = _coordinator()
        req = SimpleNamespace(
            req_pool_idx=0,
            kv=SimpleNamespace(kv_allocated_len=0),
        )
        identity = _HiSparseRequestIdentity(0, 1, id(req))
        coordinator._split_requests = {0: identity}
        coordinator._split_step_request = identity
        coordinator._split_next_request = None
        coordinator._split_pending = SimpleNamespace()
        coordinator.req_device_buffer_size = torch.tensor([1], dtype=torch.int64)
        coordinator.req_to_device_buffer = torch.tensor(
            [[3, 0, 0, 0, 0]], dtype=torch.int64
        )
        coordinator.req_to_host_pool_allocated_len = torch.tensor(
            [1], dtype=torch.int64
        )
        coordinator.req_device_buffer_tokens = torch.zeros(
            (10, 1, 5), dtype=torch.int32
        )
        coordinator.req_device_buffer_token_locs = torch.zeros(
            (10, 1, 5), dtype=torch.int32
        )
        coordinator.lru_slots = torch.zeros((10, 1, 4), dtype=torch.int16)
        coordinator._lru_init = torch.arange(4, dtype=torch.int16)
        coordinator._skip_first_backup = [False]
        coordinator.decode_producer_stream = None
        coordinator.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.zeros((1, 1), dtype=torch.int64)
        )
        coordinator.mem_pool_device = SimpleNamespace(
            translate_loc_from_full_to_compressed=lambda value: value,
            full_to_hisparse_device_index_mapping=torch.zeros(1, dtype=torch.int64),
        )
        order: list[str] = []
        coordinator.token_to_kv_pool_allocator = SimpleNamespace(
            free_hisparse_indices=lambda _value: order.append("free-device")
        )
        coordinator.mem_pool_host = SimpleNamespace(
            allocated_host_indices=lambda *_args: torch.tensor([4]),
            free=lambda _value: order.append("free-host"),
        )

        def drain(_self: object) -> None:
            order.append("drain")

        coordinator._drain_split_materialization = MethodType(drain, coordinator)
        coordinator.wait_for_pending_backup = MethodType(
            lambda _self: None, coordinator
        )

        coordinator.request_finished(req)
        coordinator.request_finished(req)

        self.assertEqual(order, ["drain", "free-device", "free-host"])
        self.assertEqual(coordinator._split_requests, {})
        self.assertIsNone(coordinator._split_pending)
        replacement = SimpleNamespace(req_pool_idx=0)
        coordinator._assert_can_activate_split_request(replacement)
        coordinator._activate_split_request(replacement)
        self.assertEqual(coordinator._split_requests[0].generation, 2)

    def test_feature_off_has_no_split_owner_state(self) -> None:
        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        self.assertFalse(coordinator.split_materialization_enabled)
        self.assertTrue(coordinator.split_worker_reusable)
        self.assertFalse(hasattr(coordinator, "_split_pending"))
        self.assertIn("_run_swap_in_kernel", coordinator.split_materialization_callable)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.version.hip is None,
        "CUDA is required for the native stream/event ownership check",
    )
    def test_device_followers_wait_for_distinct_layer_bytes(self) -> None:
        coordinator = _coordinator()
        identity = _HiSparseRequestIdentity(0, 1, 101)
        coordinator._split_requests = {0: identity}
        coordinator._split_step_request = None
        coordinator._split_next_request = identity
        device = torch.device("cuda")
        layer_count = 10
        host_token_count = 8
        row_width = 576
        physical_rows = 16
        physical_locs = [9, 7, 3, 5, 11]
        initial_tokens = [0, 1, 2, 3]

        host_layers: list[torch.Tensor] = []
        device_layers: list[torch.Tensor] = []
        for layer_id in range(layer_count):
            host = torch.empty(
                (host_token_count, 1, row_width),
                dtype=torch.bfloat16,
                device="cpu",
                pin_memory=True,
            )
            for token_id in range(host_token_count):
                host[token_id].fill_(layer_id * 16 + token_id)
            resident = torch.full(
                (physical_rows, 1, row_width),
                -321,
                dtype=torch.bfloat16,
                device=device,
            )
            for slot, token_id in enumerate(initial_tokens):
                resident[physical_locs[slot]].copy_(host[token_id], non_blocking=True)
            resident[physical_locs[4]].copy_(host[7], non_blocking=True)
            host_layers.append(host)
            device_layers.append(resident)

        coordinator.top_k = 2
        coordinator.device_buffer_size = 4
        coordinator.swap_in_block_size = 960
        coordinator.item_size_bytes = row_width * torch.bfloat16.itemsize
        coordinator.num_real_reqs = torch.tensor([1], dtype=torch.int32, device=device)
        coordinator.top_k_device_locs_buffer = torch.full(
            (1, 2), -1, dtype=torch.int32, device=device
        )
        coordinator.req_device_buffer_tokens = torch.tensor(
            [[initial_tokens + [-1]]] * layer_count,
            dtype=torch.int32,
            device=device,
        )
        coordinator.req_to_host_pool = torch.arange(
            host_token_count, dtype=torch.int64, device=device
        ).view(1, -1)
        coordinator.req_device_buffer_token_locs = torch.tensor(
            [[physical_locs]] * layer_count, dtype=torch.int32, device=device
        )
        coordinator.lru_slots = (
            torch.arange(4, dtype=torch.int16, device=device)
            .view(1, 1, -1)
            .repeat(layer_count, 1, 1)
            .contiguous()
        )
        coordinator._miss_src = torch.zeros((1, 2), dtype=torch.int64, device=device)
        coordinator._miss_dst = torch.zeros((1, 2), dtype=torch.int32, device=device)
        coordinator._miss_count = torch.zeros((1,), dtype=torch.int32, device=device)
        coordinator._prefetch_events = [torch.cuda.Event() for _ in range(3)]
        coordinator.prefetch_stream = torch.cuda.Stream()
        coordinator._prefetch_copy_blocks = 4
        coordinator.mem_pool_host = SimpleNamespace(kv_buffer=host_layers)
        coordinator.mem_pool_device = SimpleNamespace(kv_buffer=device_layers)
        coordinator.is_dsv4_hisparse = False
        coordinator.skip_io = False

        request_indices = torch.tensor([0], dtype=torch.int64, device=device)
        seq_lens = torch.tensor([host_token_count], dtype=torch.int32, device=device)
        selected_tokens = torch.tensor([[6, 7]], dtype=torch.int32, device=device)
        observed: list[torch.Tensor] = []
        tables: list[torch.Tensor] = []
        for layer_id in range(layer_count):
            if layer_id in (2, 6):
                with torch.cuda.stream(coordinator.prefetch_stream):
                    torch.cuda._sleep(2_000_000)
            table = coordinator.swap_in_selected_pages(
                request_indices, seq_lens, selected_tokens, layer_id
            )
            tables.append(table.clone())
            observed.append(device_layers[layer_id][table[0].long()].clone())

        # Queue the next plan after the final follower's selected-byte read. The
        # compute stream ordering must protect that read while reusing one table.
        coordinator._bind_split_step_request(torch.tensor([0], dtype=torch.int64))
        coordinator.swap_in_selected_pages(
            request_indices, seq_lens, selected_tokens, layer_id=0
        )
        torch.cuda.current_stream().synchronize()

        expected_table = torch.tensor([9, 11], dtype=torch.int32)
        for layer_id, (table, actual) in enumerate(zip(tables, observed)):
            self.assertTrue(torch.equal(table.cpu(), expected_table.view(1, -1)))
            expected = torch.stack([host_layers[layer_id][6], host_layers[layer_id][7]])
            self.assertTrue(
                torch.equal(actual.cpu(), expected),
                f"layer {layer_id} consumed another layer's or poisoned bytes",
            )


if __name__ == "__main__":
    unittest.main()
