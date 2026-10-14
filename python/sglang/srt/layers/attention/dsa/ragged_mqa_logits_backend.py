"""Explicit ordinary-prefill indexer selection without changing native defaults."""

from __future__ import annotations

from enum import Enum

from sglang.srt.utils import get_device_capability, is_cuda, is_hip


class DSARaggedMQALogitsBackend(Enum):
    """Choose the existing native route or an explicitly admitted SM80 fallback."""

    NATIVE = "auto"
    TORCH_SM80 = "torch-sm80"

    def is_torch_sm80(self) -> bool:
        """Return whether the ordinary-prefill SM80 fallback is selected.

        Returns
        -------
        bool
            Whether bounded Torch scoring is selected.
        """
        return self == DSARaggedMQALogitsBackend.TORCH_SM80

    @staticmethod
    def resolve(value: str) -> DSARaggedMQALogitsBackend:
        """Resolve explicit selection and validate device eligibility.

        Parameters
        ----------
        value : str
            ``auto`` preserves the existing native route; ``torch-sm80`` opts in.

        Returns
        -------
        DSARaggedMQALogitsBackend
            Frozen selection for the indexer.
        """
        if value == "auto":
            return DSARaggedMQALogitsBackend.NATIVE
        if value == "torch-sm80":
            if is_hip() or not is_cuda() or get_device_capability() != (8, 0):
                raise ValueError(
                    "dsa_ragged_mqa_logits_backend='torch-sm80' requires SM80."
                )
            return DSARaggedMQALogitsBackend.TORCH_SM80
        raise ValueError(f"Unknown dsa_ragged_mqa_logits_backend: {value!r}")

    def validate_prefill_config(
        self,
        *,
        hisparse: bool,
        speculative: bool,
        context_parallel: bool,
        paged_backend: str,
        hisparse_bf16: bool = False,
        dflash_verify: bool = False,
        dflash_offload: bool = False,
    ) -> None:
        """Reject unqualified compositions before indexer state creation.

        Parameters
        ----------
        hisparse, speculative, context_parallel : bool
            Whether offload, speculation or context parallelism is enabled.
        paged_backend : str
            Existing paired decode selector, required to be ``triton-sm80``.
        hisparse_bf16 : bool, optional
            Whether the caller admitted the ordinary BF16 HiSparse backend pair.
        dflash_verify : bool, optional
            Resolved resident DFlash admission; requires resident target storage.
        dflash_offload : bool, optional
            Separate resolved grouped BF16 offload admission; CP stays refused.
        """
        if dflash_verify and dflash_offload:
            raise ValueError(
                "DFlash storage admission must choose exactly one owner mode."
            )
        if dflash_verify and (
            not speculative or hisparse or context_parallel or not hisparse_bf16
        ):
            raise ValueError(
                "Resident DFlash requires its explicit resident BF16 recipe."
            )
        if dflash_offload and (
            not speculative or not hisparse or context_parallel or not hisparse_bf16
        ):
            raise ValueError(
                "Offload DFlash requires its explicit grouped BF16 owner recipe."
            )
        if self.is_torch_sm80() and (
            (hisparse and not hisparse_bf16)
            or (speculative and not (dflash_verify or dflash_offload))
            or context_parallel
        ):
            raise ValueError(
                "SM80 ragged logits require ordinary prefill with explicit BF16 HiSparse admission, without speculation or context parallelism."
            )
        if self.is_torch_sm80() and paged_backend != "triton-sm80":
            raise ValueError(
                "SM80 ragged logits require the paired triton-sm80 paged decode backend."
            )
