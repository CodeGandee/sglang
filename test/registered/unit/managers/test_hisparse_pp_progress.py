"""Native CPU/Gloo PP progress with declared synthetic model, events and storage.

No HiSparse overlay, replacement methods, GPUs or model weights are involved.
"""

from __future__ import annotations

import tempfile
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
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.managers import scheduler_pp_mixin as native
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=45, suite="base-a-test-cpu")


class _StopLoop(Exception):
    pass


class _Event:
    def __init__(self) -> None:
        self.synchronize_calls = 0

    def record(self, stream: object) -> None:
        pass

    def synchronize(self) -> None:
        self.synchronize_calls += 1


class _Stream:
    def wait_stream(self, stream: object) -> None:
        pass

    def wait_event(self, event: object) -> None:
        pass


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

            def requests(self) -> list[object]:
                self.visit += 1
                if self.visit == total_visits:
                    raise _StopLoop()
                return [] if pp_rank == 0 else self._pp_recv_pyobj_from_prev_stage()

            def process_input_requests(self, requests: object) -> None:
                pass

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
                batch = None
                if self.visit < populated_visits:
                    batch = SimpleNamespace(
                        reqs=[],
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
        with patch.object(
            native,
            "get_parallel",
            return_value=SimpleNamespace(
                pp_async_batch_depth=0, enable_dsa_prefill_context_parallel=False
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
        if torch.cuda.is_initialized():
            raise AssertionError("CPU progress initialized CUDA")
        (Path(directory) / f"rank-{rank}.pass").write_text(str(len(rows)))
    finally:
        dist.destroy_process_group()


class TestHiSparsePPProgress(CustomTestCase):
    """Exercise genuine PP3/PP4 loops, relays, groups and control ordering."""

    def test_native_pp3_pp4_progress_with_populated_and_empty_slots(self) -> None:
        for pp, tp, controls in (
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
