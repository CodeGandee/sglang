from __future__ import annotations

from enum import Enum

from sglang.srt.utils import get_device_capability, is_cuda, is_hip, is_sm100_supported


class DSAPagedMQALogitsBackend(Enum):
    DEEPGEMM = "deepgemm"
    CUTEDSL = "cutedsl"
    AITER = "aiter"
    TRITON_SM80 = "triton-sm80"

    def is_deepgemm(self) -> bool:
        return self == DSAPagedMQALogitsBackend.DEEPGEMM

    def is_cutedsl(self) -> bool:
        return self == DSAPagedMQALogitsBackend.CUTEDSL

    def is_aiter(self) -> bool:
        return self == DSAPagedMQALogitsBackend.AITER

    def is_triton_sm80(self) -> bool:
        """Return whether the explicit single-query SM80 kernel is selected.

        Returns
        -------
        bool
            Whether SM80 single-query Triton dispatch is selected.
        """
        return self == DSAPagedMQALogitsBackend.TRITON_SM80

    @staticmethod
    def resolve(value: str) -> DSAPagedMQALogitsBackend:
        if is_hip():
            if value not in ("auto", "aiter"):
                raise ValueError(
                    f"dsa_paged_mqa_logits_backend={value!r} is not supported on "
                    "ROCm; only 'aiter' is implemented."
                )
            return DSAPagedMQALogitsBackend.AITER

        if value == "auto" or value == "deepgemm":
            return DSAPagedMQALogitsBackend.DEEPGEMM
        if value == "triton-sm80":
            if not is_cuda() or get_device_capability() != (8, 0):
                raise ValueError(
                    "dsa_paged_mqa_logits_backend='triton-sm80' requires SM80 "
                    "(compute capability 8.0)."
                )
            return DSAPagedMQALogitsBackend.TRITON_SM80
        if value == "aiter":
            raise ValueError("dsa_paged_mqa_logits_backend='aiter' requires ROCm.")
        if value == "cutedsl":
            if not is_sm100_supported():
                raise ValueError(
                    "dsa_paged_mqa_logits_backend='cutedsl' requires SM100 (Blackwell)."
                )
            return DSAPagedMQALogitsBackend.CUTEDSL
        raise ValueError(f"Unknown dsa_paged_mqa_logits_backend: {value!r}")
