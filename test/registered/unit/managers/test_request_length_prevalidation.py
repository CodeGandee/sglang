import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestRequestLengthPrevalidation(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(
            enable_return_hidden_states=False,
            return_hidden_states_mode=None,
        )
        override.install()
        self.addCleanup(override.restore)
        self.manager = TokenizerManager.__new__(TokenizerManager)
        self.manager.context_len = 128
        self.manager.max_req_input_len = 122
        self.manager.num_reserved_tokens = 0
        self.manager.allow_auto_truncate = False
        self.manager.validate_total_tokens = False
        self.manager.is_generation = True
        self.manager.server_args = SimpleNamespace(enable_custom_logit_processor=False)
        self.manager._validate_token_ids_logprob = Mock()

    @staticmethod
    def _request(input_ids):
        return GenerateReqInput(
            input_ids=input_ids,
            sampling_params={},
            return_hidden_states=False,
        )

    def test_largest_scheduler_safe_prompt_is_accepted(self):
        input_ids = [1] * 121

        self.manager._validate_one_request(self._request(input_ids), input_ids)

    def test_first_scheduler_unsafe_prompt_is_rejected_before_queueing(self):
        input_ids = [1] * 122

        with self.assertRaisesRegex(
            ValueError,
            r"Input length \(122 tokens\) exceeds the maximum allowed length "
            r"\(122 tokens\)",
        ):
            self.manager._validate_one_request(self._request(input_ids), input_ids)


if __name__ == "__main__":
    unittest.main()
