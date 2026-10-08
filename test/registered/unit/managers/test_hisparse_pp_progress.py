"""Native CPU/Gloo PP progress with declared synthetic model, events and storage.

No HiSparse overlay, replacement methods, GPUs or model weights are involved.
"""

from __future__ import annotations

import tempfile
from array import array
import time
import unittest
from collections import Counter
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.managers import hisparse_coordinator as staging_native
from sglang.srt.managers import schedule_batch as batch_native
from sglang.srt.managers import scheduler as scheduler_native
from sglang.srt.managers import scheduler_pp_mixin as native
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


class _StopLoop(Exception):
    pass


class _Event:
    def __init__(self) -> None:
        self.synchronize_calls = 0

    def record(self, stream: object = None) -> None:
        pass

    def synchronize(self) -> None:
        self.synchronize_calls += 1

    def wait(self, stream: object) -> None:
        pass

    def query(self) -> bool:
        return True


class _Stream:
    def wait_stream(self, stream: object) -> None:
        pass

    def wait_event(self, event: object) -> None:
        pass

    def synchronize(self) -> None:
        pass


class _PendingOwners:
    """Real native request identities with declared CPU storage and DMA events.

    Native pool allocation/generations, staging admission, FIFO readiness and
    abort cleanup execute unchanged. GPU backing and prefix-cache disposal are
    bounded synthetic boundaries, independent of the PP loop fixture.
    """

    def __init__(self, tp_size: int, tp_group: object) -> None:
        self.pool = ReqToTokenPool(2, 4, "cpu", False)
        self.target = Req(
            "pending-target",
            "",
            array("q", [1, 2, 3, 4]),
            SamplingParams(max_new_tokens=16),
        )
        self.peer = Req(
            "active-peer",
            "",
            array("q", [5, 6, 7, 8]),
            SamplingParams(max_new_tokens=16),
        )
        self.pool.alloc([self.target, self.peer])
        for req in (self.target, self.peer):
            req.kv = ReqKvInfo(kv_allocated_len=4, swa_evicted_seqlen=0)
            req.kv_committed_len = 4
            req.extend_range = SimpleNamespace(end=4)
            self.pool.req_to_token[req.req_pool_idx] = (
                torch.arange(4) + req.req_pool_idx * 4
            )
        self.slot = self.target.req_pool_idx
        self.generation = int(self.pool.req_generation[self.slot])
        self.peer_slot = self.peer.req_pool_idx
        self.peer_generation = int(self.pool.req_generation[self.peer_slot])
        self.peer_core_generation = 7
        self.ready_allocations: list[Req] = []
        self.aborted: list[Req] = []
        self.queued_abort: list[AbortReq] = []
        self.device_frees: list[torch.Tensor] = []
        self.host_frees: list[torch.Tensor] = []
        owner = staging_native.HiSparseCoordinator.__new__(
            staging_native.HiSparseCoordinator
        )
        self.coordinator = owner
        owner.ack_staging_queue = []
        owner.tp_world_size, owner.tp_group = tp_size, tp_group
        owner.req_to_token_pool = self.pool
        owner.req_to_host_pool = torch.full((3, 4), -1, dtype=torch.int64)
        owner.req_to_host_pool_allocated_len = torch.zeros(3, dtype=torch.int64)
        owner._skip_first_backup = [False] * 3
        owner.write_staging_stream = _Stream()
        owner.mem_pool_device = SimpleNamespace(
            translate_loc_from_full_to_hisparse_device=lambda values: values
        )
        owner.mem_pool_host = SimpleNamespace(
            alloc_paged_token_slots=self.allocate_host,
            backup_from_device_all_layer=lambda *args, **kwargs: None,
            allocated_host_indices=lambda table, slot, length: table[
                slot, :length
            ].clone(),
            free=lambda values: self.host_frees.append(values.clone()),
        )
        owner.token_to_kv_pool_allocator = SimpleNamespace(
            free_hisparse=lambda values: self.device_frees.append(values.clone())
        )
        owner.alloc_device_buffer = self.ready_allocations.append

    def allocate_host(
        self,
        table: torch.Tensor,
        lengths: torch.Tensor,
        slot: int,
        start: int,
        end: int,
    ) -> torch.Tensor:
        values = torch.arange(start, end) + slot * 4
        table[slot, start:end] = values
        lengths[slot] = end
        return values

    def release_cached(self, req: Req, **kwargs: object) -> None:
        self.pool.free(req)
        req.kv = None

    def assert_released(self) -> None:
        if self.target.req_pool_idx is not None or self.target.kv is not None:
            raise AssertionError("native pending request slot was not released")
        if int(self.pool.req_generation[self.slot]) != self.generation:
            raise AssertionError("abort changed native request generation")
        if self.coordinator.staging_requests or self.ready_allocations:
            raise AssertionError("cancelled pending owner reached ready activation")
        if len(self.device_frees) != 1 or len(self.host_frees) != 1:
            raise AssertionError("pending backing was not freed exactly once")
        if (
            self.peer.req_pool_idx != self.peer_slot
            or int(self.pool.req_generation[self.peer_slot]) != self.peer_generation
            or self.peer_core_generation != 7
            or self.peer.to_finish is not None
        ):
            raise AssertionError("active peer identity or progress was affected")
        if self.aborted != [self.target]:
            raise AssertionError(
                "native abort acknowledgement changed request identity"
            )


def _progress_worker(
    rank: int, pp_size: int, tp_size: int, directory: str, controls: bool
) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/rendezvous",
        rank=rank,
        world_size=pp_size * tp_size,
        timeout=timedelta(seconds=15),
    )
    try:
        pp_rank, tp_rank = divmod(rank, tp_size)
        tp_groups = [
            dist.new_group(list(range(p * tp_size, (p + 1) * tp_size)))
            for p in range(pp_size)
        ]
        pp_groups = [
            dist.new_group([p * tp_size + t for p in range(pp_size)])
            for t in range(tp_size)
        ]

        def coordinator(ranks: list[int], group: object) -> GroupCoordinator:
            # Device initialization is synthetic. The real native communication
            # methods below use initialized CPU and device Gloo groups.
            obj = GroupCoordinator.__new__(GroupCoordinator)
            obj.ranks = ranks
            obj.world_size = len(ranks)
            obj.rank_in_group = ranks.index(rank)
            obj.rank = rank
            obj.cpu_group = obj.device_group = group
            obj.mq_broadcaster = obj.hpu_communicator = obj.npu_communicator = None
            obj.local_size = len(ranks)
            obj.use_symmetric_memory = use_symmetric_memory
            obj.is_allocation_symmetric = lambda: False
            return obj

        pp_group = coordinator(
            [p * tp_size + tp_rank for p in range(pp_size)], pp_groups[tp_rank]
        )
        tp_group = coordinator(
            list(range(pp_rank * tp_size, (pp_rank + 1) * tp_size)), tp_groups[pp_rank]
        )
        events: list[_Event] = []
        rows: list[tuple[str, int]] = []
        prepared_results: dict[int, object] = {}
        populated_visits, total_visits = 2 * pp_size, 4 * pp_size
        pending = (
            _PendingOwners(tp_size, tp_groups[pp_rank])
            if controls and pp_rank == 0
            else None
        )

        def event_factory() -> _Event:
            event = _Event()
            events.append(event)
            return event

        class Owner(native.SchedulerPPMixin):
            def __init__(self) -> None:
                self.ps = SimpleNamespace(
                    pp_rank=pp_rank,
                    pp_size=pp_size,
                    tp_size=tp_size,
                    attn_tp_rank=tp_rank,
                    attn_tp_size=tp_size,
                    attn_cp_rank=0,
                    attn_cp_size=1,
                    attn_dp_rank=0,
                    attn_dp_size=1,
                )
                self.pp_group, self.attn_tp_group = pp_group, tp_group
                self.attn_tp_cpu_group = tp_groups[pp_rank]
                self.world_group = SimpleNamespace(cpu_group=dist.group.WORLD)
                self.device = "cpu"
                self.device_module = SimpleNamespace(
                    Event=event_factory, current_stream=_Stream
                )
                self.forward_stream = self.copy_stream = self.schedule_stream = (
                    _Stream()
                )
                self.forward_stream_ctx = self.copy_stream_ctx = nullcontext()
                self.visit = -1
                # Startup control agreements must not shift actual loop slots.
                self._hisparse_control_ct = 4
                self.hisparse_growth_controller = (
                    SimpleNamespace(tick=self.control) if controls else None
                )
                self.future_map = SimpleNamespace(stash=self.stash)
                self.request_receiver = SimpleNamespace(recv_requests=self.requests)
                if pending is not None:
                    self.hisparse_coordinator = pending.coordinator
                    self.enable_hisparse = True
                    self.enable_hicache_storage = False
                    self.chunked_req = self.dllm_config = None
                    self.waiting_queue = []
                    self.disaggregation_mode = DisaggregationMode.NULL
                    self.grammar_manager = SimpleNamespace(
                        abort_requests=lambda req: None
                    )
                    self.tree_cache = SimpleNamespace(
                        cache_finished_req=pending.release_cached
                    )
                    self.ipc_channels = SimpleNamespace(
                        send_to_tokenizer=SimpleNamespace(
                            send_output=lambda message, req: pending.aborted.append(req)
                        )
                    )

            def requests(self) -> list[object]:
                self.visit += 1
                if self.visit == total_visits:
                    raise _StopLoop()
                rows.append(("poll", self.visit))
                if pp_rank != 0:
                    return self._pp_recv_pyobj_from_prev_stage()
                if pending is not None and pending.queued_abort:
                    if (
                        pending.target is not pending.coordinator.staging_requests[0]
                        or pending.ready_allocations
                    ):
                        raise AssertionError(
                            "abort ingress observed an activated or replaced owner"
                        )
                    rows.append(("pending-abort-poll", self.visit))
                    requests, pending.queued_abort = pending.queued_abort, []
                    return requests
                return []

            def process_input_requests(self, requests: object) -> None:
                if pending is not None:
                    for message in requests:
                        scheduler_native.Scheduler.abort_request(self, message)
                        rows.append(("pending-abort", self.visit))

            def control(self, owner: object, mb: int) -> None:
                if owner is not self or mb != self.visit % pp_size:
                    raise AssertionError(
                        "native control lost explicit microbatch identity"
                    )
                rows.append(("control", self.visit))
                dist.barrier()
                # The genuine growth controller can flush requests a second
                # time; generic completion must neither re-post nor consume output.
                self._pp_commit_comm_work(self.send_req_work)

            def get_next_batch_to_run(self, **kwargs: object) -> object:
                if pending is not None:
                    pending.coordinator.collect_ready_reqs()
                rows.append(("planner", self.visit))
                batch = None
                if self.visit < populated_visits:
                    batch = SimpleNamespace(
                        reqs=(
                            [pending.peer, pending.target]
                            if pending is not None and self.visit == 0
                            else [pending.peer]
                            if pending is not None
                            else []
                        ),
                        return_logprob=False,
                        forward_mode=ForwardMode.DECODE,
                        req_pool_indices=torch.tensor([self.visit]),
                        input_ids=object(),
                        serial=self.visit,
                    )
                return SimpleNamespace(
                    running_batch=kwargs["running_batch"], batch_to_run=batch
                )

            def run_batch(self, batch: object, proxy: object) -> GenerationBatchResult:
                rows.append(("launch", batch.serial))
                if pending is not None:
                    pending.peer.output_ids.append(batch.serial)
                return GenerationBatchResult(
                    next_token_ids=torch.full(
                        (tp_size,), 1000 + batch.serial, dtype=torch.int64
                    ),
                    pp_hidden_states_proxy_tensors=PPProxyTensors(
                        {"hidden": torch.tensor([batch.serial])}
                    ),
                    can_run_cuda_graph=False,
                )

            def stash(self, indices: torch.Tensor, payload: object) -> None:
                rows.append(("prepare", int(indices[0])))
                if int(payload.bonus_tokens[0]) != 1000 + int(indices[0]):
                    raise AssertionError("output tensor changed native target slot")

            def _pp_prep_batch_result(
                self, batch: object, metadata: object, outputs: object
            ) -> object:
                result = super()._pp_prep_batch_result(batch, metadata, outputs)
                prepared_results[batch.serial] = result
                return result

            def process_batch_result(self, batch: object, result: object) -> None:
                if result is not prepared_results.pop(batch.serial):
                    raise AssertionError("native prepared result identity was replaced")
                if result.next_token_ids.tolist() != [1000 + batch.serial] * tp_size:
                    raise AssertionError("native output lost target batch identity")
                rows.append(("consume", batch.serial))
                if pending is not None and batch.serial == 0:
                    pending.coordinator.admit_request_into_staging(pending.target)
                    pending.queued_abort.append(
                        AbortReq(rid=pending.target.rid, exact_match=True)
                    )
                    rows.append(("pending-stage", self.visit))

            def on_idle(self) -> None:
                pass

            def _pp_send_dict_to_next_stage(
                self, tensor_dict: object, **kwargs: object
            ) -> object:
                if kwargs.get("msg_type") == "output":
                    if 0 < pp_rank < pp_size - 1 and self.visit == pp_size:
                        time.sleep(0.08)
                    rows.append(("relay", int(tensor_dict["next_token_ids"][0]) - 1000))
                return super()._pp_send_dict_to_next_stage(tensor_dict, **kwargs)

        owner = Owner()
        with (
            patch.object(
                staging_native,
                "device_module",
                SimpleNamespace(Event=_Event, stream=lambda stream: nullcontext()),
            ),
            patch.object(
                batch_native,
                "get_serving",
                return_value=SimpleNamespace(strip_thinking_cache=False),
            ),
            patch.object(
                native,
                "get_parallel",
                return_value=SimpleNamespace(
                    pp_async_batch_depth=0, enable_dsa_prefill_context_parallel=False
                ),
            ),
        ):
            try:
                owner.event_loop_pp()
            except _StopLoop:
                pass
        for work in (
            owner.send_req_work,
            owner.send_proxy_work,
            owner.send_output_work,
        ):
            owner._pp_commit_comm_work(work)
        for kind in ("launch", "prepare", "consume", "relay"):
            if Counter(
                serial for operation, serial in rows if operation == kind
            ) != Counter(range(populated_visits)):
                raise AssertionError(f"{kind} was not exactly once: {rows}")
        if controls and (
            owner._hisparse_control_ct != 4 + 2 * total_visits
            or owner._pp_control_visit_id != total_visits
            or owner._hisparse_control_active
        ):
            raise AssertionError("native visit/watchdog control markers changed")
        if controls and Counter(
            serial for operation, serial in rows if operation == "control"
        ) != Counter(range(total_visits)):
            raise AssertionError("native controls missed a visit")
        if sum(event.synchronize_calls for event in events) != populated_visits or any(
            event.synchronize_calls > 1 for event in events
        ):
            raise AssertionError("native result event did not complete exactly once")
        if (
            owner.last_rank_comm_queue
            or any(owner._pp_tensor_dict_inbox.values())
            or prepared_results
        ):
            raise AssertionError("native output/frame/result remains owned after drain")
        if (
            owner._pp_control_output_relay is not None
            and not owner._pp_control_output_relay.consumed
        ):
            raise AssertionError("early output work was never adopted")
        due_poll = rows.index(("poll", pp_size - 1))
        first_consume = rows.index(("consume", 0))
        if first_consume < due_poll or first_consume < rows.index(
            ("planner", pp_size - 1)
        ):
            raise AssertionError(
                "output processing crossed the wrong native ingress boundary"
            )
        if pending is not None:
            pending.assert_released()
            polls = [serial for kind, serial in rows if kind == "pending-abort-poll"]
            aborts = [serial for kind, serial in rows if kind == "pending-abort"]
            if polls != [pp_size] or aborts != polls:
                raise AssertionError(
                    "native pending abort missed the next sole ingress poll"
                )
            staged = [serial for kind, serial in rows if kind == "pending-stage"]
            if staged != [pp_size - 1] or polls[0] != staged[0] + 1:
                raise AssertionError(
                    "native staging/planning eligibility changed visit"
                )
            if list(pending.peer.output_ids) != list(range(populated_visits)):
                raise AssertionError("active peer did not retain every model step")
        if torch.cuda.is_initialized():
            raise AssertionError("CPU progress initialized CUDA")
        (Path(directory) / f"rank-{rank}.pass").write_text(str(len(rows)))
    finally:
        dist.destroy_process_group()


class TestHiSparsePPProgress(CustomTestCase):
    """Exercise native PP2/TP2, PP3/PP4 progress and pending-owner aborts."""

    def test_native_pp3_pp4_progress_with_populated_and_empty_slots(self) -> None:
        for pp, tp, controls in (
            (2, 2, True),
            (3, 1, False),
            (3, 1, True),
            (4, 1, True),
            (4, 2, True),
        ):
            with self.subTest(pp=pp, tp=tp, controls=controls):
                with tempfile.TemporaryDirectory() as directory:
                    context = mp.spawn(
                        _progress_worker,
                        args=(pp, tp, directory, controls),
                        nprocs=pp * tp,
                        join=False,
                    )
                    deadline = time.monotonic() + 60
                    try:
                        while not context.join(timeout=1):
                            if time.monotonic() >= deadline:
                                self.fail(
                                    "native progress exceeded its finite CPU bound"
                                )
                        self.assertEqual(
                            len(list(Path(directory).glob("rank-*.pass"))), pp * tp
                        )
                    finally:
                        for process in context.processes:
                            if process.is_alive():
                                process.terminate()
                            process.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
