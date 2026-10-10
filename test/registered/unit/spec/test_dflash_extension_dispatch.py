"""CPU checks for maintained DFlash worker dispatch and PP proposal rows."""

from types import SimpleNamespace
from unittest import TestCase

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
        q_rows = torch.tensor(
            [[[0.7, 0.3], [0.8, 0.2]], [[0.6, 0.4], [0.9, 0.1]]]
        )
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
