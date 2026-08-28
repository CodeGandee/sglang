from unittest.mock import patch

from sglang.srt.utils.nvtx_utils import _torch_nvtx_range


def test_torch_nvtx_range_balances_push_and_pop():
    with (
        patch("torch.cuda.nvtx.range_push") as push,
        patch("torch.cuda.nvtx.range_pop") as pop,
        _torch_nvtx_range("scheduler.run_batch", color="red"),
    ):
        push.assert_called_once_with("scheduler.run_batch")
        pop.assert_not_called()

    pop.assert_called_once_with()
