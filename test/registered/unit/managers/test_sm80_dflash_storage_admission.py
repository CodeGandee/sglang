"""Resident, per-layer and legacy grouped DFlash admission remain distinct."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsa import sparse_mla_decode_sm80 as mla
from sglang.srt.layers.attention.dsa.ragged_mqa_logits_backend import (
    DSARaggedMQALogitsBackend,
)

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


class TestSM80DFlashStorageAdmission(unittest.TestCase):
    @staticmethod
    def _per_layer_arguments(pp_size: int = 1, tp_size: int = 8) -> SimpleNamespace:
        """Carry the frozen serving packet's values without creating CUDA state."""
        return SimpleNamespace(
            hisparse_config=json.dumps(
                {
                    "handoff": "wait",
                    "live_growth": False,
                    "top_k": 2048,
                    "device_buffer_size": 4096,
                    "host_to_device_ratio": 2,
                    "swap_in_block_size": 960,
                }
            ),
            speculative_algorithm="DFLASH",
            speculative_dflash_implementation="ours",
            speculative_num_draft_tokens=8,
            speculative_dflash_block_size=8,
            pp_size=pp_size,
            tp_size=tp_size,
            page_size=64,
            pp_async_batch_depth=0,
            disable_overlap_schedule=pp_size > 1,
            max_running_requests=2,
            context_length=16384,
            max_total_tokens=32768,
            kv_cache_dtype="bfloat16",
            dsa_decode_backend="triton-sm80",
            dsa_prefill_backend="tilelang",
            dsa_paged_mqa_logits_backend="triton-sm80",
            dsa_ragged_mqa_logits_backend="torch-sm80",
            dsa_topk_backend="torch",
            attn_cp_size=1,
            dcp_size=1,
            enable_dsa_prefill_context_parallel=False,
            enable_prefill_context_parallel=False,
            get_model_config=lambda: SimpleNamespace(
                hf_config=SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"])
            ),
        )

    def test_default_per_layer_packet_reaches_actual_native_backend_validation(
        self,
    ) -> None:
        """Admit the real default recipe, independent of legacy shared quotas."""
        from sglang.srt.arg_groups import hisparse_hook
        from sglang.srt.arg_groups.overrides import ResolvedView

        with (
            patch.object(hisparse_hook, "_is_hip", return_value=False),
            patch("sglang.srt.utils.get_device_capability", return_value=(8, 0)),
        ):
            for pp_size, tp_size in ((1, 8), (8, 1)):
                args = self._per_layer_arguments(pp_size=pp_size, tp_size=tp_size)
                self.assertEqual(args.tp_size, tp_size)
                default = json.loads(args.hisparse_config)
                for extra in (
                    {},
                    {"storage_mode": "per-layer"},
                    {
                        "storage_mode": "per-layer",
                        "initial_device_tokens": 32768,
                        "worker_budget_bytes": 1 << 30,
                        "ratio_reserve": {"8": 0.5},
                    },
                    {"ratio_reserve": {"8": 1.5}},
                ):
                    with self.subTest(pp_size=pp_size, tp_size=tp_size, extra=extra):
                        args.hisparse_config = json.dumps({**default, **extra})
                        for label in ("prefill", "decode"):
                            hisparse_hook.validate_hisparse_dsa_backend(
                                cast("ServerArgs", args),
                                "dsa_" + label + "_backend",
                                label,
                            )
                        self.assertTrue(
                            hisparse_hook._sm80_bf16_dflash_admitted(ResolvedView(args))
                        )

    def test_per_layer_refuses_legacy_quotas_and_invalid_native_recipes(self) -> None:
        """Keep fixed, synchronous, finite and ordinary-CP restrictions intact."""
        from sglang.srt.arg_groups import hisparse_hook

        config_changes: tuple[tuple[str, object], ...] = (
            ("storage_mode", "unknown"),
            ("storage_mode", "shared-window"),
            ("completion_mode", "overlapped"),
            ("shared_pool", {}),
            ("shared_pool", None),
            ("handoff", "coordinated"),
            ("live_growth", True),
            ("live_growth", 0),
            ("initial_device_tokens", 16384),
            ("initial_device_tokens", True),
            ("top_k", True),
            ("top_k", 1024),
            ("worker_budget_bytes", True),
            ("worker_budget_bytes", 0),
            ("worker_budget_bytes", -1),
            ("worker_budget_bytes", 1.5),
            ("worker_budget_bytes", "1073741824"),
        )
        argument_changes = (
            ("speculative_algorithm", "EAGLE"),
            ("speculative_dflash_implementation", "native"),
            ("speculative_num_draft_tokens", 7),
            ("speculative_dflash_block_size", 4),
            ("pp_async_batch_depth", 1),
            ("disable_overlap_schedule", True),
            ("pp_size", 0),
            ("tp_size", True),
            ("max_running_requests", 0),
            ("context_length", -1),
            ("max_total_tokens", 0),
            ("max_total_tokens", 32769),
            ("page_size", 128),
            ("attn_cp_size", 2),
            ("dcp_size", 2),
            ("enable_dsa_prefill_context_parallel", True),
            ("enable_prefill_context_parallel", True),
            ("kv_cache_dtype", "fp8_e4m3"),
            ("dsa_prefill_backend", "flashmla_sparse"),
            ("dsa_paged_mqa_logits_backend", "auto"),
            ("dsa_ragged_mqa_logits_backend", "auto"),
            ("dsa_topk_backend", "sgl-kernel"),
        )
        with (
            patch.object(hisparse_hook, "_is_hip", return_value=False),
            patch("sglang.srt.utils.get_device_capability", return_value=(8, 0)),
        ):
            for key, value in config_changes:
                args = self._per_layer_arguments()
                args.hisparse_config = json.dumps(
                    {**json.loads(args.hisparse_config), key: value}
                )
                with (
                    self.subTest(config=key, value=value),
                    self.assertRaises(ValueError),
                ):
                    hisparse_hook.validate_hisparse_dsa_backend(
                        cast("ServerArgs", args), "dsa_decode_backend", "decode"
                    )
            for key, value in argument_changes:
                args = self._per_layer_arguments()
                setattr(args, key, value)
                with (
                    self.subTest(argument=key, value=value),
                    self.assertRaises(ValueError),
                ):
                    hisparse_hook.validate_hisparse_dsa_backend(
                        cast("ServerArgs", args), "dsa_decode_backend", "decode"
                    )
            args = self._per_layer_arguments()
            legacy = json.loads(args.hisparse_config)
            legacy.update(
                storage_mode="shared-window",
                shared_pool={
                    "initial_entries": 32768,
                    "growth_entries": 32768,
                    "ceiling_entries": 32768,
                    "headroom_ratio": 1.25,
                    "total_budget_bytes": 1 << 30,
                },
            )
            args.hisparse_config = json.dumps(legacy)
            with self.assertRaises(ValueError):
                hisparse_hook.validate_hisparse_dsa_backend(
                    cast("ServerArgs", args), "dsa_decode_backend", "decode"
                )

    def test_actual_runtime_predicates_keep_per_layer_offload_distinct(self) -> None:
        """Evaluate real downstream predicates with explicit CPU config bags."""
        from sglang.srt.layers.attention.dsa import sm80_dflash_verify as verify

        args = self._per_layer_arguments()
        spec = SimpleNamespace(
            speculative_algorithm=args.speculative_algorithm,
            speculative_dflash_implementation=args.speculative_dflash_implementation,
            speculative_num_draft_tokens=args.speculative_num_draft_tokens,
            speculative_dflash_block_size=args.speculative_dflash_block_size,
        )
        kernel = SimpleNamespace(
            **{
                name: getattr(args, name)
                for name in (
                    "dsa_prefill_backend",
                    "dsa_decode_backend",
                    "dsa_paged_mqa_logits_backend",
                    "dsa_ragged_mqa_logits_backend",
                    "dsa_topk_backend",
                )
            }
        )
        memory = SimpleNamespace(enable_hisparse=True)
        with (
            patch.object(verify, "get_spec", return_value=spec),
            patch.object(
                verify,
                "get_parallel",
                return_value=SimpleNamespace(attn_cp_size=1, attn_dcp_size=1),
            ),
            patch.object(
                verify,
                "get_model",
                return_value=SimpleNamespace(kv_cache_dtype="bfloat16"),
            ),
            patch.object(
                verify, "get_exec", return_value=SimpleNamespace(kernel=kernel)
            ),
            patch.object(verify, "get_memory", return_value=memory),
        ):
            self.assertTrue(verify.sm80_dflash_offload_config_admitted())
            self.assertFalse(verify.sm80_dflash_verify_config_admitted())
            self.assertTrue(verify.sm80_dflash_any_config_admitted())
            memory.enable_hisparse = False
            self.assertFalse(verify.sm80_dflash_offload_config_admitted())
            self.assertTrue(verify.sm80_dflash_verify_config_admitted())
            memory.enable_hisparse = True
            spec.speculative_algorithm = "EAGLE"
            self.assertFalse(verify.sm80_dflash_any_config_admitted())

    def check(
        self,
        *,
        owner: str,
        hisparse: bool,
        speculative: bool,
        resident: bool,
        offload: bool,
    ) -> None:
        if owner == "sparse":
            with patch.object(mla, "_require_sm80"):
                mla.validate_sm80_sparse_mla_config(
                    device=torch.device("cuda"),
                    prefill_backend="tilelang",
                    kv_dtype=torch.bfloat16,
                    hisparse=hisparse,
                    speculative=speculative,
                    hisparse_bf16=True,
                    dflash_verify=resident,
                    dflash_offload=offload,
                )
        elif owner == "ragged":
            DSARaggedMQALogitsBackend.TORCH_SM80.validate_prefill_config(
                hisparse=hisparse,
                speculative=speculative,
                context_parallel=False,
                paged_backend="triton-sm80",
                hisparse_bf16=True,
                dflash_verify=resident,
                dflash_offload=offload,
            )
        else:
            raise AssertionError("Unknown admission owner")

    def test_ordinary_resident_and_explicit_offload_admit(self) -> None:
        for values in (
            (False, False, False, False),
            (True, False, False, False),
            (False, True, True, False),
            (True, True, False, True),
        ):
            for owner in ("sparse", "ragged"):
                with self.subTest(owner=owner, values=values):
                    self.check(
                        owner=owner,
                        hisparse=values[0],
                        speculative=values[1],
                        resident=values[2],
                        offload=values[3],
                    )

    def test_unrelated_speculation_and_wrong_storage_mode_refused(self) -> None:
        for values in (
            (True, True, False, False),
            (False, True, False, False),
            (True, True, True, False),
            (False, True, False, True),
            (True, True, True, True),
            (True, False, False, True),
        ):
            for owner in ("sparse", "ragged"):
                with (
                    self.subTest(owner=owner, values=values),
                    self.assertRaises(ValueError),
                ):
                    self.check(
                        owner=owner,
                        hisparse=values[0],
                        speculative=values[1],
                        resident=values[2],
                        offload=values[3],
                    )

    def test_actual_native_admission_live_growth_json_and_storage_refusals(
        self,
    ) -> None:
        from sglang.srt.arg_groups import hisparse_hook
        from sglang.srt.arg_groups.overrides import ResolvedView

        config = {
            "storage_mode": "shared-window",
            "top_k": 2048,
            "device_buffer_size": 4096,
            "host_to_device_ratio": 2,
            "swap_in_block_size": 960,
            "handoff": "wait",
            "live_growth": False,
            "initial_device_tokens": 16384,
            "shared_pool": {
                "initial_entries": 32768,
                "growth_entries": 32768,
                "ceiling_entries": 32768,
                "headroom_ratio": 1.25,
            },
        }
        args = SimpleNamespace(
            hisparse_config=json.dumps(config),
            speculative_algorithm="DFLASH",
            speculative_dflash_implementation="ours",
            speculative_num_draft_tokens=8,
            speculative_dflash_block_size=8,
            pp_size=1,
            tp_size=8,
            page_size=64,
            pp_async_batch_depth=0,
            disable_overlap_schedule=False,
            max_running_requests=2,
            context_length=16384,
            max_total_tokens=16384,
            kv_cache_dtype="bfloat16",
            dsa_decode_backend="triton-sm80",
            dsa_prefill_backend="tilelang",
            dsa_paged_mqa_logits_backend="triton-sm80",
            dsa_ragged_mqa_logits_backend="torch-sm80",
            dsa_topk_backend="torch",
            attn_cp_size=1,
            dcp_size=1,
            enable_dsa_prefill_context_parallel=False,
            enable_prefill_context_parallel=False,
            get_model_config=lambda: SimpleNamespace(
                hf_config=SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"])
            ),
        )
        with (
            patch.object(hisparse_hook, "_is_hip", return_value=False),
            patch("sglang.srt.utils.get_device_capability", return_value=(8, 0)),
        ):
            view = ResolvedView(args)
            self.assertTrue(hisparse_hook._sm80_bf16_dflash_admitted(view))
            for label in ("prefill", "decode"):
                hisparse_hook.validate_hisparse_dsa_backend(
                    cast("ServerArgs", args), "dsa_" + label + "_backend", label
                )
            for key, value in (
                ("live_growth", True),
                ("live_growth", 0),
                ("top_k", True),
                ("top_k", 1024),
            ):
                bad = dict(config)
                bad[key] = value
                args.hisparse_config = json.dumps(bad)
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    hisparse_hook.validate_hisparse_dsa_backend(
                        cast("ServerArgs", args), "dsa_decode_backend", "decode"
                    )
            for raw in ("[]", '{"top_k":2048,"top_k":2048}', "{bad"):
                args.hisparse_config = raw
                with self.subTest(raw=raw), self.assertRaises((ValueError, TypeError)):
                    hisparse_hook.validate_hisparse_dsa_backend(
                        cast("ServerArgs", args), "dsa_decode_backend", "decode"
                    )
            args.hisparse_config = json.dumps(config)
            args.speculative_algorithm = "EAGLE"
            with self.assertRaises(ValueError):
                hisparse_hook.validate_hisparse_dsa_backend(
                    cast("ServerArgs", args), "dsa_decode_backend", "decode"
                )

    def test_native_policy_derives_pp8_three_request_cohort(self) -> None:
        from sglang.srt.arg_groups import hisparse_hook
        from sglang.srt.arg_groups.overrides import ResolvedView

        pool = {
            "initial_entries": 49152,
            "growth_entries": 49152,
            "ceiling_entries": 49152,
            "headroom_ratio": 1.25,
        }
        config = {
            "storage_mode": "shared-window",
            "top_k": 2048,
            "handoff": "wait",
            "live_growth": False,
            "initial_device_tokens": 32768,
            "shared_pool": pool,
        }
        args = SimpleNamespace(
            hisparse_config=json.dumps(config),
            speculative_algorithm="DFLASH",
            speculative_dflash_implementation="ours",
            speculative_num_draft_tokens=8,
            speculative_dflash_block_size=8,
            pp_size=8,
            tp_size=8,
            page_size=64,
            pp_async_batch_depth=0,
            disable_overlap_schedule=True,
            max_running_requests=3,
            context_length=32768,
            max_total_tokens=32768,
        )
        self.assertTrue(hisparse_hook._sm80_bf16_dflash_admitted(ResolvedView(args)))
        pool.update(initial_entries=32768, ceiling_entries=32768, growth_entries=32768)
        args.hisparse_config = json.dumps(config)
        self.assertFalse(hisparse_hook._sm80_bf16_dflash_admitted(ResolvedView(args)))
        pool.update(initial_entries=49152, ceiling_entries=49152, growth_entries=49152)
        args.hisparse_config = json.dumps(config)
        for field, invalid in (("pp_size", 0), ("max_running_requests", True)):
            previous = getattr(args, field)
            setattr(args, field, invalid)
            self.assertFalse(
                hisparse_hook._sm80_bf16_dflash_admitted(ResolvedView(args))
            )
            setattr(args, field, previous)

    def test_native_three_request_verify_preserves_geometry_and_capacity(self) -> None:
        from sglang.srt.layers.attention.dsa import sm80_dflash_verify as verify
        from sglang.srt.model_executor.forward_batch_info import (
            ForwardBatch,
            ForwardMode,
        )
        from sglang.srt.speculative.dflash_info import DFlashVerifyInput
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        spec = DFlashVerifyInput(
            draft_token=torch.zeros(24, dtype=torch.int64),
            positions=torch.arange(24),
            draft_token_num=8,
        )
        batch = cast(
            ForwardBatch,
            SimpleNamespace(
                spec_info=spec,
                batch_size=3,
                forward_mode=ForwardMode.TARGET_VERIFY,
                spec_algorithm=SpeculativeAlgorithm.DFLASH,
                seq_lens=torch.ones(3, dtype=torch.int32),
                req_pool_indices=torch.arange(3),
                attn_cp_metadata=None,
            ),
        )
        schedule = SimpleNamespace(max_running_requests=3)
        with (
            patch.object(verify, "sm80_dflash_any_config_admitted", return_value=True),
            patch.object(verify, "get_schedule", return_value=schedule),
        ):
            self.assertEqual(
                verify.require_sm80_dflash_verify(batch, 24).query_rows, 24
            )
            schedule.max_running_requests = 2
            with self.assertRaises(NotImplementedError):
                verify.require_sm80_dflash_verify(batch, 24)
            from sglang.srt.model_executor.runner_utils import capture_mode

            with capture_mode.model_capture_mode():
                self.assertEqual(
                    verify.require_sm80_dflash_verify(batch, 24).query_rows, 24
                )
            schedule.max_running_requests = 3
            for mutate, restore in (
                (
                    lambda: setattr(spec, "draft_token_num", 4),
                    lambda: setattr(spec, "draft_token_num", 8),
                ),
                (
                    lambda: setattr(spec, "custom_mask", torch.ones(1)),
                    lambda: setattr(spec, "custom_mask", None),
                ),
            ):
                mutate()
                with self.assertRaises(NotImplementedError):
                    verify.require_sm80_dflash_verify(batch, 24)
                restore()

    def test_native_three_request_logits_delegates_exact_twenty_four_rows(self) -> None:
        from sglang.kernels.ops.attention.dsa import paged_mqa_logits as logits
        from sglang.kernels.ops.attention.dsa import (
            triton_paged_mqa_logits_sm80 as leaf,
        )

        q = torch.zeros(32, 1, 128)
        weights = torch.zeros(32, 1)
        lengths = torch.ones(24, dtype=torch.int32)
        pages = torch.zeros(24, 1, dtype=torch.int32)
        result = torch.ones(24, 64)
        with patch.object(
            leaf, "triton_paged_mqa_logits", return_value=result
        ) as compute:
            actual = logits.sm80_paged_mqa_logits(
                q,
                torch.empty(0),
                weights,
                lengths,
                pages,
                64,
                q_offset=24,
                B=3,
                next_n=8,
                is_decode=False,
                verify_layout=logits.SM80DFlashVerifyLayout(3),
            )
            self.assertIs(actual, result)
            self.assertEqual(compute.call_args.args[0].shape[0], 24)
            self.assertEqual(compute.call_args.args[2].shape[0], 24)
            compute.assert_called_once()
            for wrong in (0, True, 2):
                with self.subTest(batch=wrong), self.assertRaises(ValueError):
                    logits.sm80_paged_mqa_logits(
                        q,
                        torch.empty(0),
                        weights,
                        lengths,
                        pages,
                        64,
                        q_offset=24,
                        B=3,
                        next_n=8,
                        is_decode=False,
                        verify_layout=logits.SM80DFlashVerifyLayout(wrong),
                    )

    def test_three_request_capture_stub_has_exact_rows_without_live_cache(self) -> None:
        from sglang.kernels.ops.attention.dsa.paged_mqa_logits import (
            SM80DFlashVerifyLayout,
        )
        from sglang.srt.layers.attention import dsa_backend as backend
        from sglang.srt.layers.radix_attention import RadixAttention
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch

        config = SimpleNamespace(
            decode=SimpleNamespace(backend="breakable", bs=[1, 3]),
            prefill=SimpleNamespace(backend="disabled"),
        )
        layer = SimpleNamespace(
            tp_q_head_num=2, head_dim=576, v_head_dim=512, is_cross_attention=False
        )
        q = torch.zeros(24, 2, 512, dtype=torch.bfloat16)
        with (
            patch.object(
                backend,
                "require_sm80_dflash_verify",
                return_value=SM80DFlashVerifyLayout(3),
            ),
            patch.object(
                backend,
                "get_exec",
                return_value=SimpleNamespace(
                    graph=SimpleNamespace(cuda_graph_config=config)
                ),
            ),
        ):
            result = backend._sm80_dflash_capture_stub(
                cast(backend.DeepseekSparseAttnBackend, object()),
                q,
                None,
                None,
                cast(RadixAttention, layer),
                cast(ForwardBatch, object()),
                True,
                None,
                None,
                None,
            )
            self.assertEqual(result.shape, (24, 2, 512))
            self.assertEqual(result.dtype, torch.bfloat16)
            # Raw declarations omit B3; native capacity normalization inserts it.
            config.decode.bs = [1, 2, 4, 8]
            self.assertEqual(
                backend._sm80_dflash_capture_stub(
                    cast(backend.DeepseekSparseAttnBackend, object()),
                    q,
                    None,
                    None,
                    cast(RadixAttention, layer),
                    cast(ForwardBatch, object()),
                    True,
                    None,
                    None,
                    None,
                ).shape,
                (24, 2, 512),
            )
            config.prefill.backend = "full"
            with self.assertRaises(ValueError):
                backend._sm80_dflash_capture_stub(
                    cast(backend.DeepseekSparseAttnBackend, object()),
                    q,
                    None,
                    None,
                    cast(RadixAttention, layer),
                    cast(ForwardBatch, object()),
                    True,
                    None,
                    None,
                    None,
                )
