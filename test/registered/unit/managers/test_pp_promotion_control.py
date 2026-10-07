"""Actual native boundaries for installed HiSparse promotion integration."""

import pickle
import unittest
from array import array
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    BatchTokenizedGenerateReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.managers.scheduler_components import batch_result_processor
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _admission(rid):
    return TokenizedGenerateReqInput(
        rid=rid,
        input_text=None,
        input_ids=array("q", [1]),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=8),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
    )


class TestPPPromotionNativeBoundary(unittest.TestCase):
    def test_retiring_only_loop_polls_until_completion_then_runs_idle_once(self):
        class StopLoop(Exception):
            pass

        calls = []
        completion = SimpleNamespace(done=False)
        retained = SimpleNamespace(owned=True)
        owner = SimpleNamespace(
            ps=SimpleNamespace(pp_rank=0, pp_size=2),
            pp_group=SimpleNamespace(is_last_rank=False),
            pp_loop_size=1,
            running_mbs=[None],
            last_mbs=[None],
            mbs=[None],
            send_req_work=[],
            send_proxy_work=[],
            init_pp_loop_state=lambda: None,
            process_input_requests=lambda reqs: None,
            _pp_commit_comm_work=lambda work: None,
            _pp_send_pyobj_to_next_stage=lambda *args, **kwargs: [],
            _pp_prune_finished_hisparse_history=lambda: None,
            _pp_commit_send_output_work_and_preprocess_output_tensors=lambda *args: (
                None,
                None,
                None,
            ),
            get_next_batch_to_run=lambda **kwargs: SimpleNamespace(
                running_batch=None, batch_to_run=None
            ),
        )
        owner._pp_tick_hisparse_controls = lambda mb: (
            SchedulerPPMixin._pp_tick_hisparse_controls(owner, mb)
        )
        visits = []

        def receive():
            if len(visits) == 3:
                raise StopLoop()
            if len(visits) == 2:
                completion.done = True
            visits.append(len(visits))
            return []

        def poll(scheduler, mb, *, visit_id, certificate):
            self.assertIs(scheduler, owner)
            self.assertEqual(mb, 0)
            self.assertTrue(retained.owned)
            calls.append(("poll", visit_id, completion.done))
            if completion.done:
                retained.owned = False
                calls.append(("close", visit_id))

        owner.request_receiver = SimpleNamespace(recv_requests=receive)
        owner.hisparse_promotion_controller = SimpleNamespace(
            tick=poll,
            has_ongoing_requests=lambda: retained.owned,
        )
        owner.on_idle = lambda: calls.append(("idle", owner._hisparse_pp_visit_id))
        with (
            patch(
                "sglang.srt.managers.scheduler_pp_mixin.get_parallel",
                return_value=SimpleNamespace(pp_async_batch_depth=0),
            ),
            self.assertRaises(StopLoop),
        ):
            SchedulerPPMixin.event_loop_pp(owner)
        self.assertEqual(
            calls,
            [
                ("poll", 0, False),
                ("poll", 1, False),
                ("poll", 2, True),
                ("close", 2),
                ("idle", 2),
            ],
        )
        self.assertFalse(retained.owned)
        self.assertEqual(owner.mbs, [None])

    def test_empty_output_drains_once_in_both_slots_before_controls(self):
        class StopLoop(Exception):
            pass

        calls = []
        outputs = [(object(), None, None), (object(), None, None)]
        certificates = (object(), object())
        owner = SimpleNamespace(
            ps=SimpleNamespace(pp_rank=0, pp_size=2),
            pp_group=SimpleNamespace(is_last_rank=False),
            pp_loop_size=2,
            running_mbs=[None, None],
            last_mbs=[None, None],
            mbs=[None, None],
            send_req_work=[],
            send_proxy_work=[],
            init_pp_loop_state=lambda: None,
            process_input_requests=lambda reqs: None,
            _pp_commit_comm_work=lambda work: None,
            _pp_send_pyobj_to_next_stage=lambda *args, **kwargs: [],
            _pp_prune_finished_hisparse_history=lambda: None,
        )
        owner._pp_tick_hisparse_controls = lambda mb: (
            SchedulerPPMixin._pp_tick_hisparse_controls(owner, mb)
        )
        visits = []

        def receive():
            if len(visits) == 2:
                raise StopLoop()
            visits.append(len(visits))
            return []

        owner.request_receiver = SimpleNamespace(recv_requests=receive)
        owner._pp_commit_send_output_work_and_preprocess_output_tensors = (
            lambda first, next_mb: (
                calls.append(("drain", first, next_mb)) or outputs[first]
            )
        )

        def grow(scheduler, mb, *, promotion_visit):
            self.assertIs(scheduler, owner)
            self.assertEqual(
                (owner._hisparse_pp_mb_id, owner._hisparse_pp_visit_id),
                (mb, promotion_visit),
            )
            self.assertEqual(promotion_visit, mb)
            self.assertTrue(owner._hisparse_control_active)
            calls.append(("growth", mb))
            return certificates[mb]

        def promote(scheduler, mb, *, visit_id, certificate):
            self.assertIs(scheduler, owner)
            self.assertIs(certificate, certificates[mb])
            self.assertTrue(owner._hisparse_control_active)
            calls.append(("promotion", mb, visit_id))

        owner.hisparse_growth_controller = SimpleNamespace(tick=grow)
        owner.hisparse_promotion_controller = SimpleNamespace(
            tick=promote,
            has_ongoing_requests=lambda: False,
        )
        owner.get_next_batch_to_run = lambda **kwargs: (
            calls.append(("plan", owner._hisparse_pp_mb_id))
            or SimpleNamespace(running_batch=None, batch_to_run=None)
        )
        owner.on_idle = lambda: None
        with (
            patch(
                "sglang.srt.managers.scheduler_pp_mixin.get_parallel",
                return_value=SimpleNamespace(pp_async_batch_depth=0),
            ),
            self.assertRaises(StopLoop),
        ):
            SchedulerPPMixin.event_loop_pp(owner)
        self.assertEqual(
            calls,
            [
                ("drain", 0, 1),
                ("growth", 0),
                ("promotion", 0, 0),
                ("plan", 0),
                ("drain", 1, 0),
                ("growth", 1),
                ("promotion", 1, 1),
                ("plan", 1),
            ],
        )
        self.assertIs(owner.pp_outputs, outputs[1][0])

    def test_cpu_control_watchdog_remains_active_without_a_debug_batch(self):
        from sglang.srt.managers.scheduler_components.invariant_checker import (
            create_scheduler_watchdog,
        )

        owner = SimpleNamespace(
            is_initializing=False,
            cur_batch_for_debug=None,
            forward_ct=0,
            _hisparse_control_active=True,
            _hisparse_control_ct=1,
            _hisparse_pp_visit_id=8,
            _hisparse_pp_mb_id=0,
        )
        with patch(
            "sglang.srt.managers.scheduler_components.invariant_checker.WatchdogRaw"
        ) as raw:
            create_scheduler_watchdog(owner, watchdog_timeout=300)
        kwargs = raw.call_args.kwargs
        self.assertTrue(kwargs["is_active"]())
        self.assertEqual(kwargs["get_counter"](), 1)
        self.assertTrue(kwargs["skip_stack_dump"]())
        self.assertIn("visit=8", kwargs["dump_info"]())
        owner._hisparse_control_active = False
        owner._hisparse_control_ct += 1
        self.assertFalse(kwargs["is_active"]())
        self.assertEqual(kwargs["get_counter"](), 2)

    def test_control_watchdog_terminates_without_unbounded_external_dump(self):
        import signal

        from sglang.srt.utils.watchdog import WatchdogRaw

        raw = WatchdogRaw.__new__(WatchdogRaw)
        raw.debug_name = "CPU-control-test"
        raw.get_counter = lambda: 1
        raw.is_active = lambda: True
        raw.watchdog_timeout = 1
        raw.soft = False
        raw.dump_info = lambda: "bounded metadata"
        raw.skip_stack_dump = lambda: True
        raw.parent_process = MagicMock()
        with (
            patch("sglang.srt.utils.watchdog.time.perf_counter", side_effect=[0, 0, 2]),
            patch("sglang.srt.utils.watchdog.time.sleep"),
            patch("sglang.srt.utils.watchdog.pyspy_dump_schedulers") as dump,
        ):
            raw._watchdog_once()
        dump.assert_not_called()
        raw.parent_process.send_signal.assert_called_once_with(signal.SIGQUIT)

    def test_actual_planner_consumes_explicit_decision_and_never_local_readiness(self):
        from sglang.srt.managers.scheduler import Scheduler

        class PlanningObserved(Exception):
            pass

        owner = Scheduler.__new__(Scheduler)
        owner.process_pending_chunked_abort = lambda: None
        owner.enable_fpm = False
        owner._abort_on_waiting_timeout = lambda: None
        owner._abort_on_running_timeout = lambda batch: None
        owner.dllm_config = None
        owner.chunked_req = None
        owner.enable_hisparse = True
        owner._hisparse_pp_mb_id = 1
        owner._hisparse_pp_visit_id = 9
        pending = SimpleNamespace(hisparse_staging=True)
        batch = SimpleNamespace(
            is_empty=lambda: True, batch_is_full=False, is_prefill_only=False
        )
        owner.hisparse_coordinator = SimpleNamespace(collect_ready_reqs=MagicMock())
        owner.hisparse_promotion_controller = SimpleNamespace(
            consume=MagicMock(return_value=(pending,))
        )
        owner._build_hisparse_decode_batch = lambda reqs: batch

        def observe(running):
            self.assertIs(running, batch)
            self.assertFalse(pending.hisparse_staging)
            raise PlanningObserved()

        owner.get_new_batch_prefill = observe
        with self.assertRaises(PlanningObserved):
            owner.get_next_batch_to_run(batch, None)
        owner.hisparse_promotion_controller.consume.assert_called_once_with(1, 9)
        owner.hisparse_coordinator.collect_ready_reqs.assert_not_called()

    def test_first_stage_serializes_monotonic_episodes_with_batched_admissions(self):
        first = _admission("reuse")
        second = _admission("peer")
        owner = SimpleNamespace(
            ps=SimpleNamespace(pp_rank=0), hisparse_promotion_controller=object()
        )
        messages = [first, BatchTokenizedGenerateReqInput(batch=[second])]
        SchedulerPPMixin._pp_assign_hisparse_admissions(owner, messages)
        delivered = pickle.loads(pickle.dumps(messages))
        self.assertEqual(delivered[0].hisparse_activation_episode, 1)
        self.assertEqual(delivered[1][0].hisparse_activation_episode, 2)
        replacement = _admission("reuse")
        SchedulerPPMixin._pp_assign_hisparse_admissions(owner, [replacement])
        self.assertEqual(replacement.hisparse_activation_episode, 3)
        downstream = SimpleNamespace(
            ps=SimpleNamespace(pp_rank=1), hisparse_promotion_controller=object()
        )
        SchedulerPPMixin._pp_assign_hisparse_admissions(downstream, delivered)
        self.assertEqual(delivered[0].hisparse_activation_episode, 1)
        with self.assertRaisesRegex(RuntimeError, "ingress"):
            SchedulerPPMixin._pp_assign_hisparse_admissions(owner, [first])

    def test_unselected_ingress_leaves_episode_absent(self):
        req = _admission("ordinary")
        owner = SimpleNamespace(ps=SimpleNamespace(pp_rank=0))
        SchedulerPPMixin._pp_assign_hisparse_admissions(owner, [req])
        self.assertIsNone(req.hisparse_activation_episode)

    def test_result_origin_is_explicit_before_staging_without_mutating_rows(self):
        request = SimpleNamespace(hisparse_prefill_origin_mb=None)
        peer = SimpleNamespace(hisparse_prefill_origin_mb=1)
        rows = [request, peer]
        batch = SimpleNamespace(reqs=rows, forward_mode=ForwardMode.EXTEND)
        result = object()
        observed = []
        owner = SimpleNamespace(hisparse_promotion_controller=object())
        owner.process_batch_result = lambda b, r: observed.append(
            (b, r, tuple(req.hisparse_prefill_origin_mb for req in b.reqs))
        )
        owner._pp_prune_finished_hisparse_history = lambda: None
        SchedulerPPMixin._pp_process_batch_result(owner, batch, result, mb_id=0)
        self.assertEqual(observed, [(batch, result, (0, 1))])
        self.assertIs(batch.reqs, rows)
        with self.assertRaisesRegex(RuntimeError, "explicit origin"):
            SchedulerPPMixin._pp_process_batch_result(owner, batch, result)

    def test_prepared_pending_owner_retains_future_claim_after_backup_queue_removal(
        self,
    ):
        pending = SimpleNamespace(
            rid="A", kv=SimpleNamespace(kv_allocated_len=16384), finished=lambda: False
        )
        owner = SimpleNamespace(
            running_batch=SimpleNamespace(reqs=[pending]),
            running_mbs=[],
            mbs=[],
            chunked_req=None,
            hisparse_coordinator=SimpleNamespace(staging_requests=()),
            hisparse_promotion_controller=SimpleNamespace(
                pending_requests=lambda: (pending,)
            ),
        )
        self.assertEqual(SchedulerPPMixin._pp_future_output_requests(owner), (pending,))
        owner.running_batch.reqs = []
        self.assertEqual(SchedulerPPMixin._pp_future_output_requests(owner), (pending,))

    def test_target_space_counts_live_aliases_once_and_ignores_finished_history(self):
        live = SimpleNamespace(finished=lambda: False)
        finished = SimpleNamespace(finished=lambda: True)
        owner = SimpleNamespace(
            running_mbs=[SimpleNamespace(reqs=[live, live, finished])]
        )
        with patch(
            "sglang.srt.managers.scheduler_pp_mixin.get_parallel",
            return_value=SimpleNamespace(pp_max_micro_batch_size=2),
        ):
            self.assertEqual(SchedulerPPMixin._pp_promotion_target_space(owner, 0), 1)

    def test_history_pruning_preserves_live_peer_and_outstanding_output_alias(self):
        finished = Req.__new__(Req)
        finished.finished = lambda: True
        peer = Req.__new__(Req)
        peer.finished = lambda: False
        retired_history = ScheduleBatch(reqs=[finished, peer])
        outstanding = SimpleNamespace(reqs=[finished], filter_batch=MagicMock())
        owner = SimpleNamespace(
            last_mbs=[retired_history, outstanding], mbs=[outstanding]
        )
        # Actual ScheduleBatch filtering needs tensors for a partial filter. A
        # native finished-only history exercises its real empty fast path.
        retired_history.reqs = [finished]
        SchedulerPPMixin._pp_prune_finished_hisparse_history(owner)
        self.assertEqual(retired_history.reqs, [])
        outstanding.filter_batch.assert_not_called()
        peer_history = SimpleNamespace(reqs=[finished, peer], filter_batch=MagicMock())
        owner.last_mbs = [peer_history]
        owner.mbs = []
        SchedulerPPMixin._pp_prune_finished_hisparse_history(owner)
        peer_history.filter_batch.assert_called_once_with()


class TestPrefillStagingBatchNativeBoundary(unittest.TestCase):
    @staticmethod
    def _request(rid, *, finished=False, finish_now=False, retracted=False, chunks=0):
        req = Req.__new__(Req)
        req.rid = rid
        req.output_ids = []
        req.inflight_middle_chunks = chunks
        req.is_retracted = retracted
        req._fixture_finished = finished
        req.finished = lambda: req._fixture_finished
        req.update_finish_state = lambda: setattr(req, "_fixture_finished", finish_now)
        req.time_stats = MagicMock()
        req.return_sampling_mask = False
        req.grammar = None
        return req

    @staticmethod
    def _fixture(
        reqs,
        *,
        selected=True,
        pp_size=2,
        disaggregation=DisaggregationMode.NULL,
        fail_rid=None,
        fail_finalize=False,
    ):
        operations = []
        admitted = []
        selections = []

        def admit(req):
            operations.append(("admit", req.rid))
            if req.rid == fail_rid:
                raise RuntimeError("actual admission fixture failed")
            admitted.append(req)

        def finalize(selected_reqs):
            operations.append(("finalize", tuple(req.rid for req in selected_reqs)))
            if fail_finalize:
                raise RuntimeError("production staging finalization failed")
            selections.append(selected_reqs)

        coordinator = SimpleNamespace(
            _pp_promotion_controller=object() if selected else None,
            admit_request_into_staging=admit,
            finalize_prefill_staging_batch=finalize,
        )
        processor = SimpleNamespace(
            is_generation=True,
            server_args=SimpleNamespace(pp_size=pp_size),
            disaggregation_mode=disaggregation,
            hisparse_coordinator=coordinator,
            tree_cache=object(),
            token_to_kv_pool_allocator=SimpleNamespace(
                free_group_begin=lambda: operations.append(("begin",)),
                free_group_end=lambda: operations.append(("end",)),
            ),
            move_logprobs_to_cpu=lambda **kwargs: None,
            _validate_pp_skip_output_comm=lambda *args: None,
            _get_prefill_hidden_capture_mode=lambda *args: None,
            _maybe_update_reasoning_tokens=lambda *args: None,
            _maybe_collect_routed_experts=lambda *args: None,
            _maybe_collect_indexer_topk=lambda *args: None,
            _maybe_collect_customized_info=lambda *args: None,
            output_streamer=SimpleNamespace(
                stream_output=lambda *args: operations.append(("stream", args[2]))
            ),
            metrics_reporter=SimpleNamespace(
                report_prefill_stats=lambda **kwargs: None
            ),
        )
        processor._finalize_hisparse_prefill_staging_batch = MethodType(
            SchedulerBatchResultProcessor._finalize_hisparse_prefill_staging_batch,
            processor,
        )
        batch = SimpleNamespace(
            reqs=reqs,
            decoding_reqs=[],
            return_hidden_states=False,
            return_logprob=False,
            prefill_stats=None,
            dp_cooperation_info=None,
        )
        result = SimpleNamespace(
            copy_done=None,
            routed_experts_output=None,
            indexer_topk_output=None,
            logits_output=SimpleNamespace(hidden_states=None),
            next_token_ids=torch.arange(10, 10 + len(reqs), dtype=torch.int64),
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
            can_run_cuda_graph=True,
        )
        return processor, batch, result, operations, admitted, selections

    @staticmethod
    def _process(processor, batch, result, *, hisparse=True):
        with (
            patch.object(
                batch_result_processor,
                "get_memory",
                return_value=SimpleNamespace(enable_hisparse=hisparse),
            ),
            patch.object(batch_result_processor, "maybe_cache_unfinished_req"),
            patch.object(batch_result_processor, "release_kv_cache"),
        ):
            SchedulerBatchResultProcessor.process_batch_result_prefill(
                processor, batch, result
            )

    def test_only_successful_completed_prefills_finalize_in_native_order(self):
        a, b = self._request("a"), self._request("b")
        decoding = self._request("decoding")
        chunked = self._request("chunked", chunks=1)
        finished = self._request("finished", finished=True)
        finish_now = self._request("finish-now", finish_now=True)
        retracted = self._request("retracted", retracted=True)
        processor, batch, result, operations, admitted, selections = self._fixture(
            [a, decoding, chunked, finished, finish_now, retracted, b]
        )
        batch.decoding_reqs = [decoding]
        self._process(processor, batch, result)
        self.assertEqual(admitted, [a, b])
        self.assertEqual(selections, [[a, b]])
        self.assertEqual(
            operations[:4],
            [("begin",), ("admit", "a"), ("admit", "b"), ("finalize", ("a", "b"))],
        )
        self.assertEqual(operations[4:], [("end",), ("stream", chunked)])
        self.assertEqual(chunked.inflight_middle_chunks, 0)
        self.assertEqual(a.output_ids, [10])
        self.assertEqual(b.output_ids, [16])
        self.assertEqual(decoding.output_ids, [11])
        self.assertEqual(retracted.output_ids, [])

    def test_separate_results_do_not_merge_their_staged_lists(self):
        lists = []
        for rid in ("first", "second"):
            req = self._request(rid)
            processor, batch, result, _, _, selections = self._fixture([req])
            self._process(processor, batch, result)
            lists.append(selections[0])
            self.assertEqual(selections, [[req]])
        self.assertIsNot(lists[0], lists[1])

    def test_admission_failure_propagates_without_finalizing_partial_selection(self):
        a, b = self._request("a"), self._request("b")
        processor, batch, result, operations, admitted, selections = self._fixture(
            [a, b], fail_rid="b"
        )
        with self.assertRaisesRegex(RuntimeError, "actual admission"):
            self._process(processor, batch, result)
        self.assertEqual(admitted, [a])
        self.assertEqual(selections, [])
        self.assertEqual(operations, [("begin",), ("admit", "a"), ("admit", "b")])

    def test_unselected_single_stage_and_disaggregation_do_not_finalize(self):
        for kwargs in (
            {"selected": False},
            {"pp_size": 1},
            {"disaggregation": DisaggregationMode.PREFILL},
            {"disaggregation": DisaggregationMode.DECODE},
        ):
            with self.subTest(**kwargs):
                req = self._request("ordinary")
                processor, batch, result, _, admitted, selections = self._fixture(
                    [req], **kwargs
                )
                self._process(processor, batch, result)
                self.assertEqual(admitted, [req])
                self.assertEqual(selections, [])
                self.assertEqual(req.output_ids, [10])

    def test_product_finalization_failure_propagates_without_wait_fallback(self):
        req = self._request("a")
        processor, batch, result, operations, admitted, selections = self._fixture(
            [req], fail_finalize=True
        )
        with self.assertRaisesRegex(RuntimeError, "production staging finalization"):
            self._process(processor, batch, result)
        self.assertEqual(admitted, [req])
        self.assertEqual(selections, [])
        self.assertEqual(operations, [("begin",), ("admit", "a"), ("finalize", ("a",))])

    def test_disabled_hisparse_does_not_stage_or_finalize(self):
        req = self._request("ordinary")
        processor, batch, result, _, admitted, selections = self._fixture([req])
        self._process(processor, batch, result, hisparse=False)
        self.assertEqual(admitted, [])
        self.assertEqual(selections, [])
        self.assertEqual(req.output_ids, [10])

    def test_empty_completed_selection_is_explicit_without_fencing(self):
        chunked = self._request("chunked", chunks=1)
        processor, batch, result, _, admitted, selections = self._fixture([chunked])
        self._process(processor, batch, result)
        self.assertEqual(admitted, [])
        self.assertEqual(selections, [[]])


if __name__ == "__main__":
    unittest.main()
