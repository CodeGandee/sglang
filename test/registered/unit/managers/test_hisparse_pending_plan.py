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
    _HiSparseStagePlan,
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

    def wait_event(self, event: object) -> None:
        self.waited.append(event)
        self.operations.append("wait-event")

    def synchronize(self) -> None:
        self.operations.append("synchronize")
        self.synchronize_count += 1
        if self.fail_synchronize:
            raise RuntimeError("injected fence failure")


class _FakeEvent:
    def __init__(self, *, complete: bool = False) -> None:
        self.recorded: list[object] = []
        self.waited: list[object] = []
        self.complete = complete
        self.query_count = 0

    def record(self, stream: object) -> None:
        self.recorded.append(stream)

    def wait(self, stream: object) -> None:
        self.waited.append(stream)

    def query(self) -> bool:
        self.query_count += 1
        return self.complete


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
    @staticmethod
    def _predictive_device_coordinator() -> SimpleNamespace:
        device = torch.device("cuda")
        capacity = 2
        row_width = 576
        layer_count = 10
        host_rows = 8
        physical_rows = 16
        physical_locs = [9, 7, 3, 5, 11]
        initial_tokens = [0, 1, 2, 3]
        coordinator = _coordinator()
        host_layers: list[torch.Tensor] = []
        device_layers: list[torch.Tensor] = []
        for layer_id in range(layer_count):
            host = torch.empty(
                (host_rows, 1, row_width),
                dtype=torch.bfloat16,
                device="cpu",
                pin_memory=True,
            )
            for row in range(host_rows):
                host[row].fill_(layer_id * 16 + row)
            resident = torch.full(
                (physical_rows, 1, row_width),
                -900 - layer_id,
                dtype=torch.bfloat16,
                device=device,
            )
            host_layers.append(host)
            device_layers.append(resident)

        coordinator.device = str(device)
        coordinator.top_k = capacity
        coordinator.device_buffer_size = 4
        coordinator.swap_in_block_size = 960
        coordinator.item_size_bytes = row_width * torch.bfloat16.itemsize
        coordinator.num_real_reqs = torch.tensor([1], dtype=torch.int32, device=device)
        coordinator.top_k_device_locs_buffer = torch.full(
            (1, capacity), -1, dtype=torch.int32, device=device
        )
        token_template = torch.tensor(
            [[initial_tokens + [-1]]] * layer_count,
            dtype=torch.int32,
            device=device,
        )
        location_template = torch.tensor(
            [[physical_locs]] * layer_count,
            dtype=torch.int32,
            device=device,
        )
        lru_template = (
            torch.arange(4, dtype=torch.int16, device=device)
            .view(1, 1, -1)
            .repeat(layer_count, 1, 1)
            .contiguous()
        )
        coordinator.req_device_buffer_tokens = token_template.clone()
        coordinator.req_device_buffer_token_locs = location_template.clone()
        coordinator.lru_slots = lru_template.clone()
        coordinator.req_to_host_pool = torch.arange(
            host_rows, dtype=torch.int64, device=device
        ).view(1, -1)
        coordinator._miss_src = torch.zeros(
            (1, capacity), dtype=torch.int64, device=device
        )
        coordinator._miss_dst = torch.zeros(
            (1, capacity), dtype=torch.int32, device=device
        )
        coordinator._miss_count = torch.zeros(1, dtype=torch.int32, device=device)
        coordinator._prefetch_events = [torch.cuda.Event() for _ in range(3)]
        coordinator.prefetch_stream = torch.cuda.Stream()
        coordinator.write_staging_stream = torch.cuda.Stream()
        coordinator.decode_backup_stream = torch.cuda.Stream()
        coordinator._backup_done_event = torch.cuda.Event()
        coordinator._has_pending_backup = False
        coordinator._prefetch_copy_blocks = 4
        coordinator.mem_pool_host = SimpleNamespace(kv_buffer=host_layers)
        coordinator.mem_pool_device = SimpleNamespace(
            kv_buffer=device_layers,
            get_kv_size_bytes=lambda: sum(tensor.nbytes for tensor in device_layers),
        )
        coordinator.is_dsv4_hisparse = False
        coordinator.skip_io = False
        coordinator.enable_prefetch = True
        coordinator._overlap_enabled = False
        coordinator._enable_prediction_staging(0, predictive_overlap=True)

        request = SimpleNamespace(req_pool_idx=0, rid="predictive-device")
        native_identity = _HiSparseRequestIdentity(0, 7, id(request))
        project_identity = SimpleNamespace(
            request_id=request.rid,
            request_pool_index=0,
            generation=19,
        )
        coordinator._split_generation = native_identity.generation
        coordinator._split_requests = {0: native_identity}
        coordinator._split_step_request = native_identity
        coordinator._split_next_request = None
        coordinator._split_aborted_generation = None
        coordinator._split_step = 4
        coordinator._split_anchor_index = 2
        coordinator._split_pending = None
        coordinator._split_followers_seen = 0
        coordinator._staging_slot_next = 0
        coordinator._staging_slot_epochs = [0, 0]
        coordinator._staging_tag_sequence = 0
        coordinator._staging_slots_busy = [False, False]
        coordinator._staging_active = {}
        coordinator._staging_admitted_counts = {}
        coordinator._staging_admission_attempted = set()
        coordinator._staging_admission_events = []
        coordinator._staging_device_records = {}
        coordinator._staging_slot_stats = [
            {
                "slot": slot,
                "admissions": 0,
                "reuses": 0,
                "retirements": 0,
                "busy": False,
            }
            for slot in range(2)
        ]
        coordinator._staging_skipped_admissions = 0
        coordinator._staging_generated_publications = 0
        coordinator._staging_last_generated_positions = ()
        coordinator._staging_pending_generated_positions = ()
        coordinator._staging_identity = project_identity
        coordinator._staging_target_step = 4
        coordinator._staging_history_limit = host_rows - 1
        coordinator._staging_request_slot = 0
        coordinator._staging_observation_anchor_ids = (0, 1, 2, 6)
        coordinator._staging_views = {
            2: SimpleNamespace(
                logical_ids=torch.tensor([6, -1], dtype=torch.int32, device=device),
                valid_count=torch.tensor(1, dtype=torch.int32, device=device),
            )
        }
        request_indices = torch.tensor([0], dtype=torch.int64, device=device)
        coordinator._split_step_request_ptr = request_indices.data_ptr()
        fixture = SimpleNamespace(
            coordinator=coordinator,
            request=request,
            native_identity=native_identity,
            project_identity=project_identity,
            host_layers=host_layers,
            device_layers=device_layers,
            request_indices=request_indices,
            seq_lens=torch.tensor([host_rows], dtype=torch.int32, device=device),
            selected_tokens=torch.tensor([[6, 7]], dtype=torch.int32, device=device),
            token_template=token_template,
            location_template=location_template,
            lru_template=lru_template,
            physical_locs=physical_locs,
            initial_tokens=initial_tokens,
        )
        TestHiSparsePendingPlan._reset_predictive_device_coordinator(fixture)
        return fixture

    @staticmethod
    def _reset_predictive_device_coordinator(fixture: SimpleNamespace) -> None:
        coordinator = fixture.coordinator
        coordinator.req_device_buffer_tokens.copy_(fixture.token_template)
        coordinator.req_device_buffer_token_locs.copy_(fixture.location_template)
        coordinator.lru_slots.copy_(fixture.lru_template)
        for layer_id, resident in enumerate(fixture.device_layers):
            resident.fill_(-900 - layer_id)
            for slot, token_id in enumerate(fixture.initial_tokens):
                resident[fixture.physical_locs[slot]].copy_(
                    fixture.host_layers[layer_id][token_id], non_blocking=True
                )
            resident[fixture.physical_locs[4]].copy_(
                fixture.host_layers[layer_id][7], non_blocking=True
            )
        coordinator._split_anchor_index = 2
        coordinator._split_pending = None
        coordinator._split_followers_seen = 0
        coordinator._split_step_request_ptr = fixture.request_indices.data_ptr()
        coordinator._staging_slot_next = 0
        coordinator._staging_slots_busy[:] = [False, False]
        coordinator._staging_active.clear()
        coordinator._staging_admitted_counts.clear()
        coordinator._staging_admission_attempted.clear()
        coordinator._staging_device_records.clear()
        coordinator._staging_observation_counts.zero_()
        coordinator._staging_identity = fixture.project_identity
        coordinator._staging_target_step = 4
        coordinator._staging_request_slot = 0
        coordinator._staging_views = {
            2: SimpleNamespace(
                logical_ids=torch.tensor([6, -1], dtype=torch.int32, device="cuda"),
                valid_count=torch.tensor(1, dtype=torch.int32, device="cuda"),
            )
        }
        torch.cuda.current_stream().synchronize()

    @staticmethod
    def _warm_predictive_device_coordinator(fixture: SimpleNamespace) -> None:
        coordinator = fixture.coordinator
        prediction_ready = torch.cuda.Event()
        prediction_ready.record(torch.cuda.current_stream())
        coordinator._prediction_ready_event = prediction_ready
        plan, _, _ = coordinator._stage_prediction_rows(2)
        if plan is None:
            raise AssertionError("predictive warm-up did not acquire a stage lease")
        coordinator.swap_in_selected_pages(
            fixture.request_indices,
            fixture.seq_lens,
            fixture.selected_tokens,
            2,
        )
        for layer_id in (3, 4, 5):
            coordinator.swap_in_selected_pages(
                fixture.request_indices,
                fixture.seq_lens,
                fixture.selected_tokens,
                layer_id,
            )
        coordinator.retire_prediction_step(fixture.project_identity, 4)
        coordinator._speculative_stream.synchronize()
        coordinator.prefetch_stream.synchronize()
        torch.cuda.current_stream().synchronize()
        TestHiSparsePendingPlan._reset_predictive_device_coordinator(fixture)

    @staticmethod
    def _bindable_staging_coordinator() -> tuple[
        HiSparseCoordinator, SimpleNamespace, SimpleNamespace
    ]:
        coordinator = _coordinator()
        request = SimpleNamespace(req_pool_idx=7, rid="request-a")
        native_identity = _HiSparseRequestIdentity(7, 3, id(request))
        coordinator._split_requests = {7: native_identity}
        coordinator._split_step_request = native_identity
        coordinator._split_next_request = native_identity
        coordinator._prediction_staging_enabled = True
        coordinator.req_to_host_pool = coordinator.req_to_host_pool.repeat(8, 1)
        coordinator._staging_buffers = (
            torch.empty((2, 1, 576), dtype=torch.bfloat16),
            torch.empty((2, 1, 576), dtype=torch.bfloat16),
        )
        coordinator._staging_views = {}
        coordinator._staging_identity = None
        coordinator._staging_target_step = -1
        coordinator._staging_history_limit = 0
        coordinator._staging_request_slot = -1
        coordinator._staging_observation_anchor_ids = ()
        coordinator._staging_device_records = {}
        coordinator._staging_observation_rows = {0: 0, 1: 1, 2: 2, 6: 3}
        coordinator._staging_observation_counts = torch.zeros((4, 6), dtype=torch.int32)
        coordinator._staging_zero_count = torch.zeros(1, dtype=torch.int32)
        coordinator._staging_slots_busy = [False, False]
        coordinator._staging_slot_events = (None, None)
        coordinator._staging_slot_next = 0
        coordinator._staging_slot_epochs = [0, 0]
        coordinator._staging_active = {}
        coordinator._staging_admitted_counts = {}
        coordinator._staging_admission_attempted = set()
        coordinator._staging_tag_sequence = 0
        coordinator._staging_slot_stats = [
            {
                "slot": slot,
                "admissions": 0,
                "reuses": 0,
                "retirements": 0,
                "busy": False,
            }
            for slot in range(2)
        ]
        coordinator._staging_skipped_admissions = 0
        coordinator._staging_generated_publications = 0
        coordinator._staging_last_generated_positions = ()
        project_identity = SimpleNamespace(
            request_id="request-a", request_pool_index=7, generation=91
        )
        return coordinator, request, project_identity

    @staticmethod
    def _prediction_view(
        identity: SimpleNamespace, *, anchor: int = 2, target_step: int = 4
    ) -> SimpleNamespace:
        groups = {2: (2, 3, 4, 5), 6: (6, 7, 8, 9)}
        return SimpleNamespace(
            tag=SimpleNamespace(
                identity=identity,
                target_step=target_step,
                anchor=anchor,
                group=anchor,
                group_layers=groups.get(anchor, (anchor,)),
                representation="logical_position",
                committed_history_limit=8,
            ),
            logical_ids=torch.tensor([1, 2], dtype=torch.int32),
            valid_count=torch.tensor(2, dtype=torch.int32),
        )

    def test_prediction_binding_bridges_independent_generations_and_rejects_stale_tag(
        self,
    ) -> None:
        coordinator, request, identity = self._bindable_staging_coordinator()
        view = self._prediction_view(identity)

        coordinator.bind_prediction_staging(
            request,
            identity=identity,
            target_step=4,
            committed_history_limit=8,
            eligible_views={2: view},
        )

        self.assertEqual(coordinator._staging_identity, identity)
        self.assertEqual(coordinator._staging_observation_anchor_ids, (0, 1, 2, 6))
        self.assertNotEqual(identity.generation, 3)

        stale = self._prediction_view(identity, target_step=5)
        with self.assertRaisesRegex(RuntimeError, "stale or mis-keyed"):
            coordinator.bind_prediction_staging(
                request,
                identity=identity,
                target_step=4,
                committed_history_limit=8,
                eligible_views={2: stale},
            )

    def test_first_step_observation_covers_every_fresh_anchor(self) -> None:
        coordinator, request, identity = self._bindable_staging_coordinator()
        coordinator.bind_prediction_staging(
            request,
            identity=identity,
            target_step=0,
            committed_history_limit=0,
            eligible_views={},
        )
        coordinator.enable_prefetch = True
        fake_device = _FakeDeviceModule()
        with patch.object(hisparse_module, "device_module", fake_device):
            coordinator._synchronize_staging_observation()

        self.assertEqual(
            set(coordinator._staging_observation["anchors"]), {"0", "1", "2", "6"}
        )
        self.assertTrue(
            all(
                anchor["staged_rows"] == 0
                for anchor in coordinator._staging_observation["anchors"].values()
            )
        )

    def test_full_stage_ring_skips_without_waiting_or_losing_prediction_count(
        self,
    ) -> None:
        coordinator, request, identity = self._bindable_staging_coordinator()
        view = self._prediction_view(identity)
        coordinator.bind_prediction_staging(
            request,
            identity=identity,
            target_step=4,
            committed_history_limit=8,
            eligible_views={2: view},
        )
        coordinator._staging_slots_busy[:] = [True, True]
        coordinator._staging_slot_events = (_FakeEvent(), _FakeEvent())

        plan, eligible_count, skipped_count = coordinator._stage_prediction_rows(2)

        self.assertIsNone(plan)
        self.assertIs(eligible_count, view.valid_count)
        self.assertIs(skipped_count, view.valid_count)
        self.assertEqual(coordinator._staging_skipped_admissions, 1)

    def test_predictive_admission_uses_execution_boundaries(self) -> None:
        coordinator, request, identity = self._bindable_staging_coordinator()
        coordinator._overlap_enabled = True
        coordinator._prediction_ready_event = _FakeEvent()
        admitted: list[int] = []

        def stage_only(
            _self: object, layer_id: int
        ) -> tuple[None, torch.Tensor, torch.Tensor]:
            admitted.append(layer_id)
            zero = torch.zeros(1, dtype=torch.int32)
            return None, zero, zero

        coordinator._stage_prediction_rows = MethodType(stage_only, coordinator)
        coordinator.bind_prediction_staging(
            request,
            identity=identity,
            target_step=4,
            committed_history_limit=8,
            eligible_views={
                anchor: self._prediction_view(identity, anchor=anchor)
                for anchor in (0, 1, 2, 6)
            },
        )

        self.assertEqual(admitted, [0, 1])
        coordinator.admit_prediction_after_layer(0)
        self.assertEqual(admitted, [0, 1, 2])
        coordinator.admit_prediction_after_layer(3)
        self.assertEqual(admitted, [0, 1, 2])
        coordinator.admit_prediction_after_layer(4)
        self.assertEqual(admitted, [0, 1, 2, 6])

    def test_predictive_consumer_never_retries_a_missed_admission(self) -> None:
        coordinator, _request, identity = self._bindable_staging_coordinator()
        coordinator._overlap_enabled = True
        coordinator._staging_identity = identity
        valid = torch.tensor(2, dtype=torch.int32)
        coordinator._staging_admission_attempted = {2}
        coordinator._staging_admitted_counts = {2: (valid, valid)}

        def unexpected_stage(
            _self: object, _layer_id: int
        ) -> tuple[None, torch.Tensor, torch.Tensor]:
            raise AssertionError("consumer submitted new speculative work")

        coordinator._stage_prediction_rows = MethodType(unexpected_stage, coordinator)
        plan, eligible, skipped = coordinator._prediction_stage_for_anchor(2)

        self.assertIsNone(plan)
        self.assertIs(eligible, valid)
        self.assertIs(skipped, valid)
        with self.assertRaisesRegex(RuntimeError, "before admission"):
            coordinator._prediction_stage_for_anchor(6)

    def test_unbound_predictive_seed_requires_only_native_repair(self) -> None:
        coordinator, _request, _identity = self._bindable_staging_coordinator()
        coordinator._overlap_enabled = True
        self.assertIsNone(coordinator._staging_identity)
        with (
            patch.object(coordinator, "_stage_prediction_rows") as stage,
            patch.object(coordinator, "wait_for_pending_backup") as backup,
        ):
            for anchor in (0, 1, 2, 6):
                plan, eligible, skipped = coordinator._prediction_stage_for_anchor(
                    anchor
                )
                self.assertIsNone(plan)
                self.assertIs(eligible, coordinator._staging_zero_count)
                self.assertIs(skipped, coordinator._staging_zero_count)
            stage.assert_not_called()
            self.assertEqual(backup.call_count, 4)

    def test_scoped_retirement_joins_events_without_query_or_global_sync(self) -> None:
        coordinator, _request, identity = self._bindable_staging_coordinator()
        producer_events = (_FakeEvent(), _FakeEvent())
        reader_events = (_FakeEvent(), _FakeEvent())
        step_end = _FakeEvent()
        coordinator._overlap_enabled = True
        coordinator._overlap_nvtx_enabled = False
        coordinator._overlap_timing_events = {}
        coordinator._staging_producer_events = producer_events
        coordinator._staging_reader_events = reader_events
        coordinator._staging_step_end_event = step_end
        coordinator._staging_identity = identity
        coordinator._staging_target_step = 4
        coordinator._staging_views = {2: self._prediction_view(identity)}
        coordinator._staging_slots_busy[:] = [True, True]
        coordinator._staging_active = {2: SimpleNamespace(reader_done=reader_events[0])}
        coordinator._staging_admitted_counts = {2: (torch.tensor(1), torch.tensor(0))}
        fake_device = _FakeDeviceModule()

        with patch.object(hisparse_module, "device_module", fake_device):
            coordinator.clear_prediction_staging(identity=identity, target_step=4)

        self.assertEqual(
            fake_device.compute_stream.waited,
            [
                producer_events[0],
                reader_events[0],
                producer_events[1],
                reader_events[1],
            ],
        )
        self.assertEqual(fake_device.compute_stream.synchronize_count, 0)
        self.assertEqual(sum(event.query_count for event in producer_events), 0)
        self.assertEqual(sum(event.query_count for event in reader_events), 0)
        self.assertEqual(reader_events[0].recorded, [fake_device.compute_stream])
        self.assertEqual(step_end.recorded, [fake_device.compute_stream])
        self.assertEqual(coordinator._staging_slots_busy, [False, False])
        self.assertEqual(coordinator._staging_views, {})
        self.assertIsNone(coordinator._staging_identity)

    def test_stage_publication_sequence_survives_more_than_255_reuses(self) -> None:
        coordinator = _coordinator()
        coordinator._staging_tag_sequence = 0
        identity = _HiSparseRequestIdentity(7, 3, 101)

        tags = [
            coordinator._next_stage_tag(
                identity,
                target_step=300 + epoch,
                anchor_layer=2,
                lease_epoch=epoch,
            )
            for epoch in range(1, 301)
        ]

        self.assertEqual(len(tags), len(set(tags)))
        self.assertTrue(all(tag > 0 for tag in tags))

    def test_materialization_rejects_stale_native_generation_and_lease_epoch(
        self,
    ) -> None:
        coordinator, _request, identity = self._bindable_staging_coordinator()
        active = coordinator._split_step_request
        self.assertIsNotNone(active)
        coordinator._staging_identity = identity
        coordinator._staging_target_step = 4
        coordinator._staging_slot_epochs[0] = 301

        def plan(
            *, request: _HiSparseRequestIdentity, lease_epoch: int
        ) -> _HiSparseStagePlan:
            return _HiSparseStagePlan(
                slot=0,
                anchor_layer=2,
                logical_ids=torch.zeros(2, dtype=torch.int64),
                host_locs=torch.zeros(2, dtype=torch.int64),
                valid_count=torch.zeros((), dtype=torch.int32),
                buffer=torch.zeros((2, 1, 576), dtype=torch.bfloat16),
                step=4,
                lease_epoch=lease_epoch,
                request=request,
                project_identity=identity,
                representation="mla-bf16-width-576",
                ready_tag=torch.zeros(1, dtype=torch.int64),
                expected_tag=13,
            )

        arguments = (
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            torch.zeros((1, 2), dtype=torch.int64),
            torch.zeros((1, 2), dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            0,
        )
        stale_request = _HiSparseRequestIdentity(
            active.slot, active.generation + 1, active.object_id
        )
        with self.assertRaisesRegex(RuntimeError, "stale.*lease"):
            coordinator._materialize_staged_anchor(
                2, plan(request=stale_request, lease_epoch=301), *arguments
            )
        with self.assertRaisesRegex(RuntimeError, "stale.*lease"):
            coordinator._materialize_staged_anchor(
                2, plan(request=active, lease_epoch=300), *arguments
            )

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

    def test_cancel_drains_and_retires_busy_stage_leases(self) -> None:
        coordinator, request, identity = self._bindable_staging_coordinator()
        coordinator._staging_active = {2: SimpleNamespace()}
        coordinator._staging_device_records = {2: {"observation_row": 2}}
        coordinator._staging_slots_busy[:] = [True, True]
        coordinator._staging_slot_stats[0]["busy"] = True
        coordinator._staging_slot_stats[1]["busy"] = True
        coordinator._staging_views = {2: self._prediction_view(identity)}
        coordinator._staging_identity = identity
        coordinator._staging_target_step = 4
        coordinator._staging_history_limit = 8
        coordinator._staging_request_slot = 7
        coordinator._split_pending = SimpleNamespace()
        coordinator._overlap_enabled = True
        coordinator._speculative_stream = _FakeStream()
        coordinator._urgent_stream = _FakeStream()
        coordinator.write_staging_stream = _FakeStream()
        fake_device = _FakeDeviceModule()

        with patch.object(hisparse_module, "device_module", fake_device):
            coordinator.abort_split_materialization(safe_to_reuse=True)

        self.assertEqual(coordinator._staging_active, {})
        self.assertEqual(coordinator._staging_device_records, {})
        self.assertEqual(coordinator._staging_slots_busy, [False, False])
        self.assertEqual(
            [stats["retirements"] for stats in coordinator._staging_slot_stats],
            [1, 1],
        )
        self.assertEqual(
            [stats["busy"] for stats in coordinator._staging_slot_stats],
            [False, False],
        )
        self.assertIsNone(coordinator._staging_identity)
        self.assertTrue(coordinator.split_worker_reusable)
        self.assertEqual(coordinator.prefetch_stream.synchronize_count, 1)
        self.assertEqual(coordinator._speculative_stream.synchronize_count, 1)
        self.assertEqual(coordinator._urgent_stream.synchronize_count, 1)
        self.assertEqual(coordinator.write_staging_stream.synchronize_count, 1)

        coordinator._split_generation = 3
        coordinator._finish_split_request_release(request)
        replacement = SimpleNamespace(req_pool_idx=7)
        coordinator._assert_can_activate_split_request(replacement)
        coordinator._activate_split_request(replacement)
        self.assertEqual(coordinator._split_requests[7].generation, 4)

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

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.version.hip is None,
        "CUDA is required for the delayed prediction-stage reader check",
    )
    def test_delayed_stage_readers_block_reuse_until_completion(self) -> None:
        device = torch.device("cuda")
        capacity = 4
        row_width = 576
        layer_count = 10
        host_rows = 8
        physical_rows = 12
        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        host_layers = [
            torch.empty(
                (host_rows, 1, row_width),
                dtype=torch.bfloat16,
                pin_memory=True,
            )
            for _ in range(layer_count)
        ]
        device_layers = [
            torch.full(
                (physical_rows, 1, row_width),
                -700 - layer_id,
                dtype=torch.bfloat16,
                device=device,
            )
            for layer_id in range(layer_count)
        ]
        for layer_id, host in enumerate(host_layers):
            for row in range(host_rows):
                host[row].fill_(layer_id * 16 + row)
        native_bytes = sum(tensor.nbytes for tensor in device_layers)
        coordinator.device = str(device)
        coordinator.top_k = capacity
        coordinator.item_size_bytes = row_width * torch.bfloat16.itemsize
        coordinator.mem_pool_device = SimpleNamespace(
            kv_buffer=device_layers,
            get_kv_size_bytes=lambda: native_bytes,
        )
        coordinator.mem_pool_host = SimpleNamespace(kv_buffer=host_layers)
        coordinator._split_anchor_layers = (0, 1, 2, 6)
        coordinator._staging_slot_events = (None, None)
        coordinator._staging_slot_stats = [
            {
                "slot": slot,
                "admissions": 0,
                "reuses": 0,
                "retirements": 0,
                "busy": False,
            }
            for slot in range(2)
        ]
        coordinator._staging_slot_epochs = [0, 0]
        coordinator._staging_skipped_admissions = 0
        coordinator._enable_prediction_staging(0)
        coordinator._staging_slots_busy = [False, False]
        coordinator._staging_active = {}
        coordinator._staging_device_records = {}
        coordinator._staging_generated_publications = 0
        coordinator._staging_last_generated_positions = ()
        coordinator.req_to_host_pool = torch.arange(
            host_rows, dtype=torch.int64, device=device
        ).view(1, -1)
        native_identity = _HiSparseRequestIdentity(0, 7, 101)
        project_identity = SimpleNamespace(
            request_id="device-reader", request_pool_index=0, generation=19
        )
        coordinator._split_requests = {0: native_identity}
        coordinator._split_step_request = native_identity
        coordinator._staging_identity = project_identity
        coordinator._staging_target_step = 4
        coordinator._staging_history_limit = 7
        coordinator._staging_request_slot = 0
        coordinator._staging_slot_next = 0
        coordinator._staging_observation_anchor_ids = (0, 1, 2, 6)
        coordinator._staging_views = {
            layer_id: SimpleNamespace(
                logical_ids=torch.tensor(
                    [layer_id, -1, -1, -1], dtype=torch.int32, device=device
                ),
                valid_count=torch.tensor(1, dtype=torch.int32, device=device),
            )
            for layer_id in (0, 1, 2)
        }
        miss_src = torch.tensor([[0, 0, 0, 0]], dtype=torch.int64, device=device)
        second_miss_src = torch.tensor([[1, 0, 0, 0]], dtype=torch.int64, device=device)
        miss_dst = torch.tensor([[5, 0, 0, 0]], dtype=torch.int32, device=device)
        miss_count = torch.tensor([1], dtype=torch.int32, device=device)
        delayed = torch.cuda.Stream()
        delayed.wait_stream(torch.cuda.current_stream())
        allocated_bytes = coordinator.staging_allocation_receipt["total_bytes"]

        with torch.cuda.stream(delayed):
            absent, absent_eligible, absent_skipped = (
                coordinator._stage_prediction_rows(6)
            )
            self.assertIsNone(absent)
            coordinator._materialize_staged_anchor(
                6,
                absent,
                absent_eligible,
                absent_skipped,
                miss_src,
                miss_dst,
                miss_count,
                0,
            )
            first, first_eligible, first_skipped = coordinator._stage_prediction_rows(0)
            self.assertIsNotNone(first)
            torch.cuda._sleep(500_000_000)
            coordinator._materialize_staged_anchor(
                0,
                first,
                first_eligible,
                first_skipped,
                miss_src,
                miss_dst,
                miss_count,
                0,
            )
            second, second_eligible, second_skipped = (
                coordinator._stage_prediction_rows(1)
            )
            self.assertIsNotNone(second)
            torch.cuda._sleep(500_000_000)
            coordinator._materialize_staged_anchor(
                1,
                second,
                second_eligible,
                second_skipped,
                second_miss_src,
                miss_dst,
                miss_count,
                0,
            )
            blocked, _, _ = coordinator._stage_prediction_rows(2)

        self.assertIsNone(blocked)
        self.assertEqual(coordinator._staging_skipped_admissions, 1)
        self.assertEqual(
            coordinator.staging_allocation_receipt["total_bytes"], allocated_bytes
        )
        delayed.synchronize()
        self.assertTrue(torch.equal(device_layers[0][5].cpu(), host_layers[0][0]))
        self.assertTrue(torch.equal(device_layers[1][5].cpu(), host_layers[1][1]))
        self.assertTrue(torch.equal(device_layers[6][5].cpu(), host_layers[6][0]))

        reused, _, _ = coordinator._stage_prediction_rows(2)
        self.assertIsNotNone(reused)
        torch.cuda.current_stream().synchronize()
        runtime = coordinator.staging_runtime_receipt
        self.assertIsNotNone(runtime)
        self.assertGreaterEqual(sum(row["reuses"] for row in runtime["slot_stats"]), 1)
        self.assertGreaterEqual(
            sum(row["retirements"] for row in runtime["slot_stats"]), 1
        )

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.version.hip is None,
        "CUDA is required for predictive urgent-independence coverage",
    )
    def test_predictive_urgent_repair_and_native_followers_progress_while_stage_is_delayed(
        self,
    ) -> None:
        fixture = self._predictive_device_coordinator()
        self._warm_predictive_device_coordinator(fixture)
        coordinator = fixture.coordinator

        delayed = torch.cuda.Stream()
        gate = torch.cuda.Event()
        with torch.cuda.stream(delayed):
            torch.cuda._sleep(100_000_000)
            gate.record(delayed)
        delayed_tail = torch.cuda.Event()
        delayed_tail.record(delayed)

        coordinator._prediction_ready_event = gate
        staged, _, _ = coordinator._stage_prediction_rows(2)
        self.assertIsNotNone(staged)

        real_table = coordinator.swap_in_selected_pages(
            fixture.request_indices,
            fixture.seq_lens,
            fixture.selected_tokens,
            2,
        )
        tables = {2: real_table.clone()}
        observed = {
            2: fixture.device_layers[2][real_table[0].long()].clone(),
        }
        for layer_id in (3, 4, 5):
            table = coordinator.swap_in_selected_pages(
                fixture.request_indices,
                fixture.seq_lens,
                fixture.selected_tokens,
                layer_id,
            )
            tables[layer_id] = table.clone()
            observed[layer_id] = fixture.device_layers[layer_id][
                table[0].long()
            ].clone()
        urgent_done = coordinator._urgent_done_event
        if urgent_done is None:
            raise AssertionError("predictive coordinator did not create U event")
        urgent_done.synchronize()
        for layer_id in (3, 4, 5):
            coordinator._prefetch_events[
                coordinator._prefetch_slot[layer_id]
            ].synchronize()
        self.assertTrue(urgent_done.query())
        self.assertFalse(delayed_tail.query())

        expected_table = torch.tensor([9, 11], dtype=torch.int32, device="cuda")
        for layer_id in (2, 3, 4, 5):
            self.assertTrue(torch.equal(tables[layer_id][0], expected_table))
            expected = torch.stack(
                [fixture.host_layers[layer_id][6], fixture.host_layers[layer_id][7]]
            )
            self.assertTrue(
                torch.equal(observed[layer_id].cpu(), expected),
                f"layer {layer_id} consumed another layer's or poisoned bytes",
            )

        coordinator.retire_prediction_step(fixture.project_identity, 4)
        torch.cuda.current_stream().synchronize()
        delayed_tail.synchronize()
        self.assertEqual(coordinator._staging_slots_busy, [False, False])
        self.assertGreaterEqual(
            sum(stats["retirements"] for stats in coordinator._staging_slot_stats),
            1,
        )

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.version.hip is None,
        "CUDA is required for predictive cancellation coverage",
    )
    def test_predictive_cancellation_drains_owned_streams_before_slot_reuse(
        self,
    ) -> None:
        fixture = self._predictive_device_coordinator()
        self._warm_predictive_device_coordinator(fixture)
        coordinator = fixture.coordinator
        sentinel_targets = {
            name: torch.full((256,), -1, dtype=torch.int32, device="cuda")
            for name in ("C", "U", "N", "P", "backup", "write_staging")
        }
        stream_by_name = {
            "C": torch.cuda.current_stream(),
            "U": coordinator._urgent_stream,
            "N": coordinator.prefetch_stream,
            "P": coordinator._speculative_stream,
            "backup": coordinator.decode_backup_stream,
            "write_staging": coordinator.write_staging_stream,
        }
        completion_events: dict[str, torch.cuda.Event] = {}
        for name, stream in stream_by_name.items():
            event = (
                coordinator._backup_done_event
                if name == "backup"
                else torch.cuda.Event(enable_timing=False)
            )
            with torch.cuda.stream(stream):
                sentinel_targets[name].fill_(0)
                event.record(stream)
            completion_events[name] = event
        for stream in stream_by_name.values():
            stream.synchronize()

        prediction_ready = torch.cuda.Event()
        prediction_ready.record(torch.cuda.current_stream())
        coordinator._prediction_ready_event = prediction_ready
        staged, _, _ = coordinator._stage_prediction_rows(2)
        self.assertIsNotNone(staged)

        for name, stream in stream_by_name.items():
            event = completion_events[name]
            with torch.cuda.stream(stream):
                torch.cuda._sleep(100_000_000)
                sentinel_targets[name].fill_(17)
                event.record(stream)
        coordinator._has_pending_backup = True

        for name, event in completion_events.items():
            self.assertFalse(event.query(), f"{name} stream completed before abort")
        coordinator.abort_split_materialization(safe_to_reuse=True)
        for event in completion_events.values():
            self.assertTrue(event.query())
        self.assertEqual(coordinator._staging_slots_busy, [False, False])
        self.assertEqual(coordinator._staging_active, {})
        self.assertTrue(coordinator.split_worker_reusable)

        for target in sentinel_targets.values():
            target.fill_(29)
        torch.cuda.current_stream().synchronize()
        for name, target in sentinel_targets.items():
            self.assertTrue(
                torch.equal(target, torch.full_like(target, 29)),
                f"{name} stream wrote after cancellation reuse",
            )

        replacement = SimpleNamespace(req_pool_idx=0)
        coordinator._split_generation = fixture.native_identity.generation
        coordinator._split_requests = {}
        coordinator._assert_can_activate_split_request(replacement)
        coordinator._activate_split_request(replacement)
        self.assertEqual(
            coordinator._split_requests[0].generation,
            fixture.native_identity.generation + 1,
        )


if __name__ == "__main__":
    unittest.main()
