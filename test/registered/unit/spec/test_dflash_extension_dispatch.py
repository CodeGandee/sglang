"""CPU checks for maintained DFlash worker dispatch and PP proposal rows."""

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch

from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def _draft(owners: tuple[tuple[str, int], ...]) -> DFlashDraftInputV2:
    rows = len(owners)
    value = DFlashDraftInputV2(
        topk_p=torch.empty((rows, 0)),
        topk_index=torch.empty((rows, 0), dtype=torch.int64),
        bonus_tokens=torch.arange(rows, dtype=torch.int64),
        new_seq_lens=torch.arange(rows, dtype=torch.int64) + 10,
        hidden_states=torch.empty((rows, 0)),
    )
    value.pp_episode_keys = owners
    value.pp_draft_tokens = torch.arange(rows * 3).reshape(rows, 3)
    value.pp_prefix_lens = value.new_seq_lens.clone()
    value.pp_round_ids = tuple(range(rows))
    value.pp_anchor_ids = tuple(int(value) for value in value.pp_draft_tokens[:, 0])
    value.pp_force_fallback = (False,) * rows
    value.pp_selector_candidate_ids = torch.arange(rows * 2).reshape(rows, 1, 2)
    value.pp_selector_q_rows = torch.ones((rows, 1, 2))
    return value


class TestDFlashExtensionDispatch(TestCase):
    def test_omitted_selector_uses_maintained_worker(self) -> None:
        from predkv_extensions.dflash2.worker import DFlashWorker

        selected = SpeculativeAlgorithm.DFLASH.create_worker(SimpleNamespace())
        self.assertIs(selected, DFlashWorker)

    def test_explicit_original_selector_uses_native_worker(self) -> None:
        args = SimpleNamespace(speculative_dflash_implementation="original")
        selected = SpeculativeAlgorithm.DFLASH.create_worker(args)
        self.assertIs(selected, DFlashWorkerV2)

    def test_filter_and_merge_preserve_candidate_rows(self) -> None:
        first = _draft((("a", 1), ("b", 1)))
        first.filter_batch(torch.tensor([1]), new_indices_cpu=[1])
        self.assertEqual(first.pp_episode_keys, (("b", 1),))
        self.assertEqual(first.pp_round_ids, (1,))
        self.assertEqual(first.pp_draft_tokens.tolist(), [[3, 4, 5]])
        self.assertEqual(first.pp_selector_candidate_ids.tolist(), [[[2, 3]]])
        second = _draft((("c", 2),))
        first.merge_batch(second)
        self.assertEqual(first.pp_episode_keys, (("b", 1), ("c", 2)))
        self.assertEqual(first.pp_draft_tokens.shape, (2, 3))
        self.assertEqual(first.pp_selector_candidate_ids.shape, (2, 1, 2))

    def test_non_greedy_acceptance_preserves_selector_distributions(self) -> None:
        from predkv_extensions.dflash2.worker import DFlashWorker

        candidates = torch.tensor([[1, 2, 3], [4, 5, 6]])
        candidate_ids = torch.tensor([[[2, 7], [3, 8]], [[5, 9], [6, 10]]])
        q_rows = torch.tensor([[[0.7, 0.3], [0.8, 0.2]], [[0.6, 0.4], [0.9, 0.1]]])
        captured = []

        def sampled(**kwargs):
            captured.append(kwargs)
            return torch.tensor([1, 0], dtype=torch.int32), torch.tensor([7, 9])

        worker = SimpleNamespace(block_size=3, _selector_sampling_accept=sampled)
        native_input = SimpleNamespace(
            pp_selector_candidate_ids=candidate_ids,
            pp_selector_q_rows=q_rows,
        )
        accept, commit, bonus, out, frontiers, _ = DFlashWorker._accept_block(
            worker,
            candidates=candidates,
            next_token_logits=torch.zeros((6, 16)),
            sampling_info=SimpleNamespace(is_all_greedy=False),
            draft_input=native_input,
            prefix_lens=torch.tensor([20, 30]),
            bs=2,
        )
        self.assertIs(captured[0]["candidate_ids"], candidate_ids)
        self.assertIs(captured[0]["q_rows"], q_rows)
        self.assertEqual(accept.tolist(), [1, 0])
        self.assertEqual(commit.tolist(), [2, 1])
        self.assertEqual(bonus.tolist(), [7, 9])
        self.assertEqual(out.tolist(), [[2, 7, 0], [9, 6, 0]])
        self.assertEqual(frontiers.tolist(), [22, 31])


class TestDFlashConnectedLifecycle(TestCase):
    def _worker_batch(self):
        from predkv_extensions.dflash2.state import DFlashRequestState
        from predkv_extensions.dflash2.worker import DFlashWorker
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        worker = object.__new__(DFlashWorker)
        worker.device = "cpu"
        worker.block_size = 3
        worker._mask_token_id = 99
        worker._request_state = DFlashRequestState()
        worker._request_slots = {}
        worker._request_refs = {}
        worker._verified_prefixes = {}
        worker._capacity_errors = {}
        worker._locally_published_decisions = set()
        worker._pp_active = True
        worker._pp_is_last = True
        retired, committed = [], []
        worker.target_binding = SimpleNamespace(
            retire_request=retired.append,
            commit_verified=lambda reqs, frontiers, decision_id, **kwargs: (
                committed.append(
                    (
                        tuple(req.rid for req in reqs),
                        list(frontiers),
                        decision_id,
                        tuple(req.rid for req in kwargs.get("rejected_requests", ())),
                    )
                )
            ),
        )
        reqs = [
            SimpleNamespace(
                rid=rid,
                dflash_request_epoch=1,
                req_pool_idx=slot,
                output_ids=[anchor],
                origin_input_ids=[anchor],
                finished=lambda: False,
            )
            for rid, slot, anchor in (("a", 4, 0), ("b", 6, 3))
        ]
        pool = SimpleNamespace(req_generation=[1] * 16)
        batch = SimpleNamespace(
            reqs=reqs,
            req_to_token_pool=pool,
            forward_mode=ForwardMode.DECODE,
            is_extend_in_batch=False,
            seq_lens=torch.tensor([10, 11]),
            seq_lens_cpu=torch.tensor([10, 11]),
            req_pool_indices=torch.tensor([4, 6]),
            return_logprob=False,
        )
        worker._sync_request_state(batch)
        return worker, batch, retired, committed

    def test_native_filter_reorder_and_sampling_rows_follow_episode(self):
        worker, batch, _, _ = self._worker_batch()
        draft = _draft((("a", 1), ("b", 1)))
        draft._pp_draft_tokens = draft.pp_draft_tokens
        worker._pp_tag_draft_owners(draft, batch)
        draft.filter_batch(torch.tensor([1, 0]), new_indices_cpu=[1, 0])
        batch.reqs.reverse()
        batch.seq_lens = batch.seq_lens.flip(0)
        batch.seq_lens_cpu = batch.seq_lens_cpu.flip(0)
        worker._sync_request_state(batch)
        block, fallback = worker._pp_prepare_live_block(draft, batch)
        self.assertEqual(block.tolist(), [[3, 4, 5], [0, 1, 2]])
        self.assertEqual(fallback, ())
        self.assertEqual(draft.pp_selector_candidate_ids.tolist(), [[[2, 3]], [[0, 1]]])
        # Consumption retires each exact request's stored native row once.
        self.assertIsNone(
            worker._request_state.take_native(
                worker._request_slots[next(iter(worker._request_slots))]
            )
        )

    def test_retired_slot_and_repeated_rid_cannot_reuse_old_proposal(self):
        worker, batch, retired, _ = self._worker_batch()
        draft = _draft((("a", 1), ("b", 1)))
        draft._pp_draft_tokens = draft.pp_draft_tokens
        worker._pp_tag_draft_owners(draft, batch)
        old = batch.reqs[0]
        old.req_pool_idx = None
        batch.req_to_token_pool.req_generation[4] = 2
        replacement = SimpleNamespace(
            rid="a",
            dflash_request_epoch=2,
            req_pool_idx=4,
            output_ids=[8],
            origin_input_ids=[8],
            finished=lambda: False,
        )
        batch.reqs[0] = replacement
        worker._sync_request_state(batch)
        block, fallback = worker._pp_prepare_live_block(draft, batch)
        self.assertEqual(block.tolist(), [[8, 99, 99], [3, 4, 5]])
        self.assertEqual(fallback, (0,))
        self.assertEqual(retired, [])
        self.assertNotIn(old, worker._request_refs.values())

    def test_capacity_commit_preserves_peer_and_duplicate_decision_rejects(self):
        from predkv_extensions.dflash2.state import RequestEpisode

        worker, batch, _, committed = self._worker_batch()
        worker._capacity_errors = {RequestEpisode("a", 1): "page ceiling"}
        commit, out, frontier = (
            torch.tensor([2, 1]),
            torch.ones((2, 3), dtype=torch.int64),
            torch.tensor([12, 12]),
        )
        worker.mask_target_outcomes(batch, batch.seq_lens, commit, out, frontier)
        self.assertEqual(commit.tolist(), [0, 1])
        self.assertEqual(frontier.tolist(), [10, 12])
        worker._commit_target_decision(batch, frontier, 7)
        self.assertEqual(committed, [(("b",), [12], 7, ("a",))])
        self.assertEqual(batch.reqs[0]._dflash_round, 0)
        self.assertEqual(batch.reqs[1]._dflash_round, 1)
        with self.assertRaisesRegex(ValueError, "decision"):
            worker._apply_decision(batch, frontier, 7)

    def test_native_proposal_uses_folded_graph_sampler_and_compact_cache(self):
        worker = object.__new__(DFlashWorkerV2)
        worker.device = "cpu"
        worker.block_size = 3
        worker._mask_token_id = 99
        worker._noise_embed_scale = 1.0
        worker._use_triton_prepare_block = False
        worker.use_compact_draft_cache = True
        worker._block_pos_offsets = torch.arange(3)
        worker._draft_block_ids_buf = torch.empty((2, 3), dtype=torch.int64)
        worker._draft_block_positions_buf = torch.empty((2, 3), dtype=torch.int64)
        worker._draft_block_tokens_buf = torch.empty((2, 3), dtype=torch.int64)
        worker._draft_verify_out_cache_loc_buf = torch.empty((2, 3), dtype=torch.int64)
        worker._draft_block_end_buf = torch.empty(2)
        worker._draft_seq_lens_cpu_buf = torch.empty(2, dtype=torch.int32)
        worker._ensure_draft_block_buffers = lambda bs: None
        worker._compute_compact_draft_seq_lens = lambda prefix: prefix - 4
        rebuilt = []
        worker._rebuild_compact_draft_cache = lambda **kw: rebuilt.append(kw)
        worker._fill_compact_seq_lens_cpu_bound = lambda **kw: kw["out"].copy_(
            torch.tensor([9, 10])
        )
        staged = []
        worker._draft_sampler = SimpleNamespace(
            out=torch.tensor([2, 3, 5, 6]),
            candidate_out=torch.ones((2, 2, 2), dtype=torch.int64),
            q_out=torch.full((2, 2, 2), 0.5),
            stage_sampling_params=lambda **kw: staged.append(kw),
        )
        worker.selector = object()
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                model=SimpleNamespace(
                    get_input_embeddings=lambda: (
                        lambda tokens: torch.zeros((*tokens.shape, 4))
                    ),
                    lm_head=SimpleNamespace(weight=torch.empty(16, 4)),
                )
            )
        )
        seen = []
        worker.draft_model_runner = SimpleNamespace(
            tp_group=None,
            forward=lambda value: (
                seen.append(value)
                or SimpleNamespace(
                    logits_output=SimpleNamespace(hidden_states=None),
                    can_run_graph=True,
                )
            ),
        )
        worker.model_runner = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.zeros((8, 32), dtype=torch.int64)
            )
        )
        batch = SimpleNamespace(
            seq_lens_cpu=torch.tensor([10, 11]),
            req_pool_indices=torch.tensor([4, 6]),
            sampling_info=SimpleNamespace(is_all_greedy=False),
        )
        draft = SimpleNamespace(bonus_tokens=torch.tensor([1, 4]), nxt_kv_lens_cpu=None)
        worker._draft_block_spec_info = object()
        with patch(
            "sglang.srt.speculative.dflash_worker_v2.assign_extend_cache_locs_func",
            return_value=torch.arange(6),
        ):
            block, _, _, _ = worker.propose_block(
                batch, draft, prefix_lens=torch.tensor([10, 11])
            )
        self.assertEqual(block.tolist(), [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(seen[0].seq_lens.tolist(), [6, 7])
        self.assertEqual(len(rebuilt), 1)
        self.assertEqual(len(staged), 1)
        self.assertEqual(
            worker._selector_sample[1].data_ptr(),
            worker._draft_sampler.q_out.data_ptr(),
        )

    def test_native_result_settles_healthy_tokens_without_mutating_failed_prefix(self):
        from sglang.srt.managers.scheduler_components.batch_result_processor import (
            SchedulerBatchResultProcessor,
        )
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.managers.utils import GenerationBatchResult
        from sglang.srt.sampling.sampling_params import SamplingParams

        params = SamplingParams(max_new_tokens=8, temperature=0)
        params.normalize(None)
        failed = Req("failed", "", [1, 2], params)
        healthy = Req("healthy", "", [1, 2], params)
        for req in (failed, healthy):
            req.dflash_request_epoch = 1
            req.kv_committed_len = 10
        result = GenerationBatchResult(
            next_token_ids=torch.tensor([0, 0, 0, 4, 5, 0]),
            accept_lens=torch.tensor([0, 2]),
            speculative_num_draft_tokens=3,
            request_capacity_errors=(("failed", 1, "page ceiling"),),
        )
        processor = SimpleNamespace(
            model_worker=SimpleNamespace(on_verify_complete_cpu=lambda *a, **kw: None),
            advance_grammar_fsm=lambda *a: None,
        )
        tokens = SchedulerBatchResultProcessor._resolve_spec_v2_tokens(
            processor, result, SimpleNamespace(reqs=[failed, healthy])
        )
        self.assertEqual(tokens, [[], [4, 5]])
        self.assertEqual((failed.kv_committed_len, healthy.kv_committed_len), (10, 12))
        self.assertEqual((failed.spec_verify_ct, healthy.spec_verify_ct), (0, 1))
        self.assertEqual(result.num_correct_drafts_per_req_cpu, [0, 1])

    def test_native_rebind_constructs_maintained_result_abi(self):
        worker, batch, _, _ = self._worker_batch()
        worker._pp_is_last = False
        worker._decision_id = 0
        values = {
            "next_token_ids": torch.tensor([10, 0, 0, 20, 0, 0]),
            "dflash_new_seq_lens": torch.tensor([11, 12]),
            "dflash_accept_lens": torch.tensor([1, 1]),
            "dflash_decision_id": 7,
        }
        result = worker.rebind_pp_result(
            None,
            batch,
            SimpleNamespace(can_run_cuda_graph=False),
            SimpleNamespace(tensors=values),
        )
        self.assertEqual(result.accept_lens.tolist(), [1, 1])
        self.assertEqual(result.new_seq_lens.tolist(), [11, 12])
        with self.assertRaisesRegex(ValueError, "decision"):
            worker.rebind_pp_result(
                None,
                batch,
                SimpleNamespace(can_run_cuda_graph=False),
                SimpleNamespace(tensors=values),
            )

    def test_real_finished_handler_cleans_target_once_then_dflash_state(self):
        from predkv_extensions.dflash2.state import RequestEpisode
        from sglang.srt.managers.schedule_batch import FINISH_ABORT
        from sglang.srt.managers.scheduler_components.batch_result_processor import (
            SchedulerBatchResultProcessor,
        )

        worker, batch, _, _ = self._worker_batch()
        request = batch.reqs[0]
        request.finished = lambda: True
        request.finished_reason = FINISH_ABORT()
        request.multimodal_inputs = None
        request.session = None
        request.time_stats = SimpleNamespace(set_completion_time=lambda: None)
        batch.mamba_track_mask_cpu = None
        cleanups = []

        def cleanup(req):
            if cleanups:
                raise ValueError("target request already retired")
            cleanups.append(req)

        worker.target_binding = SimpleNamespace(retire_request=cleanup)
        processor = SimpleNamespace(
            _mamba_prefix_cache_update=lambda *a, **kw: None,
            _maybe_collect_routed_experts=lambda *a: None,
            _maybe_collect_indexer_topk=lambda *a: None,
            _maybe_collect_customized_info=lambda *a: None,
            draft_worker=None,
            model_worker=worker,
            hisparse_coordinator=SimpleNamespace(request_finished=cleanup),
            tree_cache=object(),
        )
        module = "sglang.srt.managers.scheduler_components.batch_result_processor"
        with (
            patch(module + ".mamba_extra_buffer_lazy_enabled", return_value=False),
            patch(
                module + ".get_disagg",
                return_value=SimpleNamespace(
                    disaggregation_decode_enable_offload_kvcache=False
                ),
            ),
            patch(
                module + ".get_memory",
                return_value=SimpleNamespace(enable_hisparse=True),
            ),
            patch(module + ".release_kv_cache") as release,
        ):
            SchedulerBatchResultProcessor._handle_finish_state_updated_req(
                processor, request, batch, None, 0, None
            )
        self.assertEqual(cleanups, [request])
        release.assert_called_once()
        self.assertFalse(worker._request_state.contains(RequestEpisode("a", 1)))
        self.assertTrue(worker._request_state.contains(RequestEpisode("b", 1)))
