import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.models.llama import LlamaForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = PPMissingLayer()
        self.start_layer = 8
        self.end_layer = 16


class _DummyHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros((4, 4)))


class TestLlamaPipelineParallel(CustomTestCase):
    def test_last_rank_materializes_and_loads_tied_lm_head(self) -> None:
        pp_group = SimpleNamespace(world_size=2, is_last_rank=True)
        parallel = SimpleNamespace(enable_dp_lm_head=False)
        config = SimpleNamespace(
            tie_word_embeddings=True,
            vocab_size=4,
            hidden_size=4,
        )
        head = _DummyHead()
        with (
            patch("sglang.srt.models.llama.get_pp_group", return_value=pp_group),
            patch("sglang.srt.models.llama.get_parallel", return_value=parallel),
            patch.object(LlamaForCausalLM, "_init_model", return_value=_DummyModel()),
            patch("sglang.srt.models.llama.ParallelLMHead", return_value=head),
            patch(
                "sglang.srt.models.llama.LogitsProcessor",
                return_value=torch.nn.Identity(),
            ),
            patch("sglang.srt.models.llama.Pooler", return_value=torch.nn.Identity()),
        ):
            model = LlamaForCausalLM(config)

        self.assertIs(model.lm_head, head)
        loaded = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        model._legacy_load_weights((("model.embed_tokens.weight", loaded),))
        torch.testing.assert_close(model.lm_head.weight, loaded)


if __name__ == "__main__":
    unittest.main()
