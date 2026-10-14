"""Resolved SM80 BF16 DFlash recipe and explicit storage-mode admission."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.kernels.ops.attention.dsa.paged_mqa_logits import (
    SM80DFlashVerifyLayout,
)
from sglang.srt.runtime_context import (
    get_exec,
    get_memory,
    get_model,
    get_parallel,
    get_schedule,
    get_spec,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def _sm80_dflash_common_recipe() -> bool:
    """Identify the common BF16 recipe for resident and grouped target storage.

    Returns
    -------
    bool
        Whether resolved native configuration matches DFlash/ours block eight,
        the BF16 TileLang/SM80/Torch recipe, and execution without CP.
        Separate storage predicates establish resident or grouped ownership.
    """
    spec = get_spec()
    parallel = get_parallel()
    kernel = get_exec().kernel
    return bool(
        spec.speculative_algorithm == "DFLASH"
        and spec.speculative_dflash_implementation == "ours"
        and spec.speculative_num_draft_tokens == 8
        and spec.speculative_dflash_block_size in (None, 8)
        and parallel.attn_cp_size == 1
        and parallel.attn_dcp_size == 1
        and get_model().kv_cache_dtype == "bfloat16"
        and kernel.dsa_prefill_backend == "tilelang"
        and kernel.dsa_decode_backend == "triton-sm80"
        and kernel.dsa_paged_mqa_logits_backend == "triton-sm80"
        and kernel.dsa_ragged_mqa_logits_backend == "torch-sm80"
        and kernel.dsa_topk_backend == "torch"
    )


def sm80_dflash_verify_config_admitted() -> bool:
    """Return whether the resolved common recipe uses resident target storage."""
    return _sm80_dflash_common_recipe() and not get_memory().enable_hisparse


def sm80_dflash_offload_config_admitted() -> bool:
    """Return whether BF16 offload additionally requires a live grouped target owner.

    Notes
    -----
    Native HiSparse admission and product policy validate fixed-wait bounds.
    The attention call separately verifies the genuine target transaction.
    """
    return _sm80_dflash_common_recipe() and get_memory().enable_hisparse


def sm80_dflash_any_config_admitted() -> bool:
    """Return whether either explicitly admitted target storage mode is selected."""
    return sm80_dflash_verify_config_admitted() or sm80_dflash_offload_config_admitted()


def require_sm80_dflash_verify(
    forward_batch: ForwardBatch, query_rows: int | None = None
) -> SM80DFlashVerifyLayout:
    """Validate real target input and return its call-local query layout.

    Parameters
    ----------
    forward_batch : ForwardBatch
        Actual maintained input to target verification.
    query_rows : int, optional
        Available query rows including native padding.

    Returns
    -------
    SM80DFlashVerifyLayout
        Validated unpadded request-major layout for the current call.
    """
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.runner_utils import capture_mode
    from sglang.srt.speculative.dflash_info import DFlashVerifyInput
    from sglang.srt.speculative.spec_info import SpecInputType, SpeculativeAlgorithm

    spec = forward_batch.spec_info
    batch_size = forward_batch.batch_size
    if (
        not sm80_dflash_any_config_admitted()
        or forward_batch.forward_mode != ForwardMode.TARGET_VERIFY
        or forward_batch.spec_algorithm != SpeculativeAlgorithm.DFLASH
        or not isinstance(spec, DFlashVerifyInput)
        or spec.spec_input_type != SpecInputType.DFLASH_VERIFY
        or spec.draft_token_num != 8
        or spec.topk != 1
        or spec.custom_mask is not None
        or spec.ragged_verify_layout is not None
        or forward_batch.attn_cp_metadata is not None
        or type(batch_size) is not int
        or batch_size < 1
        or not capture_mode.is_capture_mode
        and batch_size > get_schedule().max_running_requests
        or forward_batch.seq_lens.shape != (batch_size,)
        or forward_batch.req_pool_indices.shape != (batch_size,)
        or query_rows is not None
        and query_rows < batch_size * 8
    ):
        raise NotImplementedError(
            "Resident SM80 DFlash admission requires linear request-major Bx8 target verification without CP"
        )
    return SM80DFlashVerifyLayout(batch_size=batch_size)
