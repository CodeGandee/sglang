"""Native PP loop ordering and all-microbatch request-accounting regressions."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _StopLoop(Exception):
    pass


class TestPPLiveGrowthControl(unittest.TestCase):
    def test_active_output_tuple_is_consumed_once_after_current_batch_launch(self):
        calls = []
        old_target = SimpleNamespace(reqs=[SimpleNamespace(rid="A")])
        new_current = SimpleNamespace(reqs=[SimpleNamespace(rid="B")])
        output_proxy = object()
        old_result = object()
        launch_event = object()
        input_proxy = object()

        class CopyEvent:
            def synchronize(self):
                calls.append("d2h")

        copy_event = CopyEvent()
        outputs = (output_proxy, old_result, copy_event)
        scheduler = SimpleNamespace(
            ps=SimpleNamespace(pp_rank=0, pp_size=2),
            pp_group=SimpleNamespace(is_last_rank=False),
            pp_loop_size=2,
            running_mbs=[None, None],
            last_mbs=[None, old_target],
            mbs=[None, old_target],
            send_req_work=[],
            send_proxy_work=[],
            mb_metadata=[object(), object()],
            last_rank_comm_queue=[],
        )
        scheduler._pp_tick_hisparse_controls = lambda mb: (
            SchedulerPPMixin._pp_tick_hisparse_controls(scheduler, mb)
        )
        scheduler.init_pp_loop_state = lambda: None
        scheduler.process_input_requests = lambda reqs: calls.append("requests")
        scheduler._pp_commit_comm_work = lambda work: None
        scheduler._pp_send_pyobj_to_next_stage = lambda data, async_send=False: []

        def receive():
            if calls:
                raise _StopLoop()
            return []

        scheduler.request_receiver = SimpleNamespace(recv_requests=receive)

        def drain(next_first, next_mb):
            calls.append("output")
            self.assertEqual((next_first, next_mb), (0, 1))
            self.assertIs(scheduler.mbs[next_mb], old_target)
            return outputs

        scheduler._pp_commit_send_output_work_and_preprocess_output_tensors = drain
        scheduler.hisparse_growth_controller = SimpleNamespace(
            tick=lambda owner, mb, **kwargs: calls.append("growth")
        )

        scheduler.hisparse_promotion_controller = SimpleNamespace(
            tick=lambda owner, mb, visit_id, certificate: calls.append("promotion"),
            has_ongoing_requests=lambda: False,
        )
        scheduler._pp_prune_finished_hisparse_history = lambda: None

        def plan(**kwargs):
            calls.append("plan")
            return SimpleNamespace(running_batch=new_current, batch_to_run=new_current)

        scheduler.get_next_batch_to_run = plan

        def receive_proxy():
            calls.append("proxy")
            return input_proxy

        scheduler._pp_recv_proxy_tensors = receive_proxy

        def launch(mb, current, proxy, metadata, queued):
            calls.append("launch")
            self.assertEqual(mb, 0)
            self.assertIs(current, new_current)
            self.assertIs(proxy, input_proxy)
            return (
                SimpleNamespace(
                    pp_hidden_states_proxy_tensors=SimpleNamespace(tensors={})
                ),
                launch_event,
            )

        scheduler._pp_launch_batch = launch

        def process(batch, result, *, mb_id=None):
            calls.append("process")
            self.assertIs(batch, old_target)
            self.assertIs(result, old_result)
            # Prefill completion can create new pending ownership AFTER the
            # immediate empty plan. No current-visit proof covers this result.
            calls.append("late-registration")

        scheduler._pp_process_batch_result = process

        def wait_launch(event):
            calls.append("wait-launch")
            self.assertIs(event, launch_event)

        scheduler.device_module = SimpleNamespace(
            current_stream=lambda: SimpleNamespace(wait_event=wait_launch)
        )
        scheduler._pp_send_dict_to_next_stage = lambda tensors, **kwargs: (
            calls.append("send-proxy") or []
        )
        with (
            patch(
                "sglang.srt.managers.scheduler_pp_mixin.get_parallel",
                return_value=SimpleNamespace(pp_async_batch_depth=0),
            ),
            self.assertRaises(_StopLoop),
        ):
            SchedulerPPMixin.event_loop_pp(scheduler)
        self.assertEqual(
            calls,
            [
                "requests",
                "output",
                "growth",
                "promotion",
                "plan",
                "proxy",
                "launch",
                "d2h",
                "process",
                "late-registration",
                "wait-launch",
                "send-proxy",
            ],
        )
        self.assertIs(scheduler.pp_outputs, output_proxy)
        self.assertIs(scheduler.last_mbs[1], old_target)

    def test_next_boundary_reactivates_full_protocol_after_post_plan_registration(self):
        pending = []
        calls = []
        proof = object()
        owner = SimpleNamespace(_pp_prune_finished_hisparse_history=lambda: None)

        def growth(scheduler, mb, *, promotion_visit):
            calls.append(("growth", mb, promotion_visit, tuple(pending)))
            return None if pending else proof

        def promotion(scheduler, mb, *, visit_id, certificate):
            calls.append(("promotion", mb, visit_id, certificate))

        owner.hisparse_growth_controller = SimpleNamespace(tick=growth)
        owner.hisparse_promotion_controller = SimpleNamespace(tick=promotion)
        SchedulerPPMixin._pp_tick_hisparse_controls(owner, 0)
        # This mutation models the actual later _pp_process_batch_result owner;
        # the helper must obtain a NEW visit proof, never retain the old token.
        pending.append("new-exact-owner")
        SchedulerPPMixin._pp_tick_hisparse_controls(owner, 1)
        self.assertEqual(
            calls,
            [
                ("growth", 0, 0, ()),
                ("promotion", 0, 0, proof),
                ("growth", 1, 1, ("new-exact-owner",)),
                ("promotion", 1, 1, None),
            ],
        )

    def test_rank_zero_drains_native_output_once_before_growth_control(self):
        calls = []
        outputs = (object(), object(), None)
        scheduler = SimpleNamespace()
        scheduler.ps = SimpleNamespace(pp_rank=0, pp_size=2)
        scheduler.pp_group = SimpleNamespace(is_last_rank=False)
        scheduler.pp_loop_size = 2
        scheduler.running_mbs = [None, None]
        scheduler.last_mbs = [None, None]
        scheduler.mbs = [None, None]
        scheduler.send_req_work = []
        scheduler.send_proxy_work = []
        scheduler._pp_tick_hisparse_controls = lambda mb: (
            SchedulerPPMixin._pp_tick_hisparse_controls(scheduler, mb)
        )
        scheduler.init_pp_loop_state = lambda: None
        scheduler.process_input_requests = lambda reqs: calls.append("requests")
        scheduler._pp_commit_comm_work = lambda work: None
        scheduler._pp_send_pyobj_to_next_stage = lambda data, async_send=False: []

        def receive():
            if calls:
                raise _StopLoop()
            return []

        scheduler.request_receiver = SimpleNamespace(recv_requests=receive)

        def drain(next_first, next_mb):
            calls.append("output")
            self.assertEqual((next_first, next_mb), (0, 1))
            return outputs

        scheduler._pp_commit_send_output_work_and_preprocess_output_tensors = drain
        scheduler.hisparse_growth_controller = SimpleNamespace(
            tick=lambda owner, mb, **kwargs: calls.append("growth")
        )

        def plan(**kwargs):
            calls.append("plan")
            return SimpleNamespace(running_batch=None, batch_to_run=None)

        scheduler.get_next_batch_to_run = plan
        with (
            patch(
                "sglang.srt.managers.scheduler_pp_mixin.get_parallel",
                return_value=SimpleNamespace(pp_async_batch_depth=0),
            ),
            self.assertRaises(_StopLoop),
        ):
            SchedulerPPMixin.event_loop_pp(scheduler)
        self.assertEqual(calls, ["requests", "output", "growth", "plan"])
        self.assertIs(scheduler.pp_outputs, outputs[0])

    def test_all_unfinished_allocated_requests_are_deduplicated_across_mbs_and_staging(
        self,
    ):
        def req(rid, allocated, finished=False):
            return SimpleNamespace(
                rid=rid,
                kv=SimpleNamespace(kv_allocated_len=allocated),
                finished=lambda: finished,
            )

        a, b, chunk = req("A", 10), req("B", 20), req("chunk", 5)
        finished, waiting = req("finished", 30, True), req("waiting", 0)
        batch = lambda requests: SimpleNamespace(reqs=requests)
        scheduler = SimpleNamespace(
            running_batch=batch([a]),
            running_mbs=[batch([a, b])],
            mbs=[batch([b, finished, waiting])],
            cur_batch_for_debug=batch([a]),
            hisparse_coordinator=SimpleNamespace(staging_requests=(b, chunk)),
            chunked_req=chunk,
        )
        actual = SchedulerPPMixin._pp_future_output_requests(scheduler, batch([a]))
        self.assertEqual([request.rid for request in actual], ["A", "B", "chunk"])


if __name__ == "__main__":
    unittest.main()
