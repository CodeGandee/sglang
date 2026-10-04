"""Protect first-chunk output claims before subsequent PP batch admission."""

import unittest

import test_prefill_adder as _fixture_module
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPPAlignedChunkReservation(unittest.TestCase):
    def test_new_aligned_truncated_chunk_reserves_outputs_before_next_request(self):
        fixture = _fixture_module.TestPrefillAdder()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        fixture.mock_token_allocator.available_size.return_value = 768
        adder = fixture.create_adder(
            fixture.create_running_batch(),
            page_size=64,
            rem_chunk_tokens=256,
            future_requests=(),
        )
        first = fixture._create_delayer_req(300)
        first.rid = "C"
        first.sampling_params.max_new_tokens = 128
        result = adder.add_one_req(
            first, has_chunked_req=False, truncation_align_size=192
        )
        self.assertEqual(result, AddReqResult.CONTINUE)
        self.assertEqual(first.extend_range.length, 192)
        self.assertEqual(adder.rem_chunk_tokens, 64)
        # One committed chunk, one alignment page and the entire remaining
        # output claim. The independent expected demand is 192 + 64 + 128.
        self.assertEqual(adder.rem_total_token_offset, 384)
        second = fixture._create_delayer_req(32)
        second.rid = "D"
        second.sampling_params.max_new_tokens = 320
        self.assertEqual(
            adder.add_one_req(
                second, has_chunked_req=False, truncation_align_size=None
            ),
            AddReqResult.NO_TOKEN,
        )
        self.assertEqual(adder.can_run_list, [first])


if __name__ == "__main__":
    unittest.main()
