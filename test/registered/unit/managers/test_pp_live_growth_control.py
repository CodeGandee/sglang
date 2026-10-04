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
            tick=lambda owner, mb: calls.append("growth")
        )

        def plan(**kwargs):
            calls.append("plan")
            return SimpleNamespace(running_batch=None, batch_to_run=None)

        scheduler.get_next_batch_to_run = plan
        with patch(
            "sglang.srt.managers.scheduler_pp_mixin.get_parallel",
            return_value=SimpleNamespace(pp_async_batch_depth=0),
        ):
            with self.assertRaises(_StopLoop):
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
