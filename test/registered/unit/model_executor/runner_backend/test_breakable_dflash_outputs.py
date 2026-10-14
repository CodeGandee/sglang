"""CPU contracts for token-row graph outputs and independent DFlash roles.

Native owners execute against CPU tensors. Graph construction, replay, device
timers and model execution are explicit fixtures; no CUDA qualification follows.
"""

import unittest
from contextlib import nullcontext
from dataclasses import replace
from types import MethodType, SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers import aux_hidden_states as auxiliary
from sglang.srt.layers.attention import dsa_backend as attention
from sglang.srt.layers.attention.base_attn_backend import SharedReadEnds
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.model_executor.runner import decode_cuda_graph_runner as decode
from sglang.srt.model_executor.runner.base_cuda_graph_runner import BaseCudaGraphRunner
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend import breakable_cuda_graph_backend as bcg
from sglang.srt.model_executor.runner_backend import utils as resolution
from sglang.srt.speculative import dflash_utils
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestBreakableDFlashOutputs(unittest.TestCase):
    """Exercise actual candidate output owners without constructing a model."""

    def setUp(self) -> None:
        self.backend = bcg.BreakableCudaGraphBackend.__new__(
            bcg.BreakableCudaGraphBackend
        )
        self.backend._cuda_graph_runner = cast(
            BaseCudaGraphRunner,
            SimpleNamespace(output_buffer_rows=lambda key: key.size * 8),
        )
        self.backend._device_module = SimpleNamespace(synchronize=Mock())
        self.backend._tp_group = SimpleNamespace(barrier=Mock())
        self.backend._debug_eager = False
        self.backend._shared_output_buffer = None
        self.backend._pool = None
        self.backend._capture_stream = None
        self.backend.deduped_cuda_graph = None
        self.backend._graphs = {}
        self.backend._outputs = {}
        self.backend._capture_inputs = {}
        self.backend._aux_packers = {}

    def test_request_bucket_capacity_preserves_token_width(self) -> None:
        owner = SimpleNamespace(ragged_verify_mode=False, captured_req_width=8)
        self.assertEqual(
            decode.DecodeCudaGraphRunner.output_buffer_rows(
                cast(decode.DecodeCudaGraphRunner, owner), ShapeKey(2)
            ),
            16,
        )
        owner.ragged_verify_mode = True
        self.assertEqual(
            decode.DecodeCudaGraphRunner.output_buffer_rows(
                cast(decode.DecodeCudaGraphRunner, owner), ShapeKey(13)
            ),
            13,
        )
        self.assertEqual(
            BaseCudaGraphRunner.output_buffer_rows(
                cast(BaseCudaGraphRunner, owner), ShapeKey(13)
            ),
            13,
        )

    def test_actual_capture_keeps_eight_rows_and_input_ownership(self) -> None:
        output = LogitsProcessorOutput(
            torch.arange(40).reshape(8, 5), hidden_states=torch.ones(8, 3)
        )
        forward = Mock(return_value=output)
        inputs = object()
        with (
            patch.object(bcg, "BreakableCUDAGraph", return_value=Mock()),
            patch.object(bcg, "BreakableCUDAGraphCapture", return_value=nullcontext()),
            patch.object(bcg, "graph_pool_capture_scope", return_value=nullcontext()),
        ):
            self.backend.capture_one(ShapeKey(1), forward, inputs)
        self.assertEqual(forward.call_count, 3)
        saved = self.backend._outputs[ShapeKey(1)]
        self.assertTrue(torch.equal(saved.next_token_logits, output.next_token_logits))
        self.assertEqual(saved.hidden_states.shape, (8, 3))
        self.assertIs(self.backend._capture_inputs[ShapeKey(1)], inputs)

    def test_changed_logits_refresh_same_backing(self) -> None:
        output = LogitsProcessorOutput(
            torch.zeros(16, 7), hidden_states=torch.ones(16, 3)
        )
        storage = self.backend._alloc_full_buffer(output, 16)
        pointer = storage.next_token_logits.data_ptr()
        self.backend._copy_output_to_buffer(output, storage, 16)
        output.next_token_logits.fill_(9)
        self.backend._copy_output_to_buffer(output, storage, 8)
        view = self.backend._slice_output(storage, 8)
        self.assertEqual(view.next_token_logits.data_ptr(), pointer)
        self.assertEqual(view.next_token_logits.shape, (8, 7))
        self.assertTrue(torch.equal(view.next_token_logits, torch.full((8, 7), 9.0)))

    def test_only_captured_aux_owner_publishes_after_successful_replay(self) -> None:
        packers: list[auxiliary.AuxHiddenStatePacker] = []

        def forward() -> PPProxyTensors:
            packer = auxiliary.AuxHiddenStatePacker(1)
            packer.append(torch.full((8, 3), float(len(packers))))
            packers.append(packer)
            return PPProxyTensors({"hidden_states": packer.finalize()})

        graph = Mock()
        with (
            patch.object(bcg, "BreakableCUDAGraph", return_value=graph),
            patch.object(bcg, "BreakableCUDAGraphCapture", return_value=nullcontext()),
            patch.object(bcg, "graph_pool_capture_scope", return_value=nullcontext()),
        ):
            self.backend.capture_one(ShapeKey(1), forward)
        self.assertEqual(len(packers), 3)
        self.assertEqual(self.backend._aux_packers[ShapeKey(1)], (packers[2],))
        observed: list[torch.Tensor] = []
        graph.replay.side_effect = lambda: packers[2].finalize().fill_(9)
        with (
            patch.object(
                auxiliary,
                "_aux_replay_observer",
                lambda p: observed.append(p.finalize().clone()),
            ),
            patch.object(bcg, "graph_pool_replay_scope", side_effect=nullcontext),
        ):
            self.backend.replay(ShapeKey(1), None)
            self.assertEqual(len(observed), 1)
            self.assertTrue(torch.equal(observed[0], torch.full((8, 3), 9.0)))
            failure = RuntimeError("synthetic replay failure")
            graph.replay.side_effect = failure
            with self.assertRaises(RuntimeError) as raised:
                self.backend.replay(ShapeKey(1), None)
            self.assertIs(raised.exception, failure)
            self.assertEqual(len(observed), 1)
        with patch.object(self.backend, "close"):
            self.backend.cleanup()
        self.assertEqual(self.backend._aux_packers, {})

    def test_hidden_only_draft_and_full_logits_keep_structure(self) -> None:
        for output in (
            LogitsProcessorOutput(None, hidden_states=torch.ones(8, 3)),
            LogitsProcessorOutput(None, full_logits=torch.ones(8, 5)),
        ):
            with self.subTest(output=output):
                storage = self.backend._alloc_full_buffer(output, 8)
                self.backend._copy_output_to_buffer(output, storage, 8)
                self.assertEqual(self.backend._output_rows(storage, 8), 8)
                self.assertIsNone(storage.next_token_logits)

    def test_unknown_fields_and_mismatched_rows_refuse(self) -> None:
        base = LogitsProcessorOutput(torch.ones(8, 5), hidden_states=torch.ones(8, 3))
        for output in (
            replace(base, customized_info={"foreign": []}),
            replace(base, next_token_logprobs=torch.ones(8)),
            replace(base, hidden_states=torch.ones(1, 3)),
            replace(base, next_token_logits=torch.ones(8)),
            LogitsProcessorOutput(None),
        ):
            with self.subTest(output=output), self.assertRaises(ValueError):
                self.backend._alloc_full_buffer(output, 8)

    def test_copy_rejects_representation_or_presence_changes(self) -> None:
        output = LogitsProcessorOutput(torch.ones(8, 5), hidden_states=torch.ones(8, 3))
        for changed in (
            replace(output, next_token_logits=torch.ones(8, 1)),
            replace(output, next_token_logits=torch.ones(8, 5, dtype=torch.bfloat16)),
            replace(output, hidden_states=None),
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.backend._copy_output_to_buffer(
                    changed, self.backend._alloc_full_buffer(output, 8), 8
                )

    def test_proxy_fields_preserve_token_rows_and_refuse_changed_keys(self) -> None:
        output = PPProxyTensors(
            {
                "hidden_states": torch.ones(8, 3),
                "dflash_aux_l19": torch.full((8, 3), 2.0),
            }
        )
        storage = self.backend._alloc_full_buffer(output, 8)
        self.backend._copy_output_to_buffer(output, storage, 8)
        view = self.backend._slice_output(storage, 8)
        self.assertEqual(set(view.tensors), set(output.tensors))
        for name, value in output.tensors.items():
            self.assertTrue(torch.equal(view.tensors[name], value))
        with self.assertRaises(ValueError):
            self.backend._copy_output_to_buffer(
                PPProxyTensors({"hidden_states": output.tensors["hidden_states"]}),
                storage,
                8,
            )

    def test_native_execute_returns_all_verification_proxy_rows(self) -> None:
        output = PPProxyTensors({"hidden_states": torch.arange(24).reshape(8, 3)})
        runner = SimpleNamespace(
            model_runner=SimpleNamespace(device_timer=None),
            backend=SimpleNamespace(
                replay_session=nullcontext, replay=Mock(return_value=output)
            ),
            _replay_attn_backend=lambda: None,
            _resolve_shared_read_ends=lambda *args: SharedReadEnds.UNKNOWN,
            load_batch=Mock(),
            _replay_graph_key=ShapeKey(1),
            bs=1,
            raw_num_token=8,
        )
        with (
            patch.object(decode, "device_timer_ctx", return_value=nullcontext()),
            patch.object(
                decode.envs.SGLANG_LOG_DECODE_GRAPH_KEY, "get", return_value=False
            ),
        ):
            result = decode.DecodeCudaGraphRunner.execute(
                cast(decode.DecodeCudaGraphRunner, runner),
                SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY),
            )
        self.assertTrue(
            torch.equal(
                result.tensors["hidden_states"], output.tensors["hidden_states"]
            )
        )
        runner.backend.replay.assert_called_once()

    def test_draft_backend_is_independent_of_shared_target_config(self) -> None:
        config = SimpleNamespace(decode=SimpleNamespace(backend="breakable"))
        args = SimpleNamespace(
            cuda_graph_config=config, enable_memory_saver=False, debug_cuda_graph=False
        )
        model = SimpleNamespace(
            server_args=args,
            device="cuda",
            is_draft_worker=False,
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
        )
        runner = SimpleNamespace(model_runner=model)
        with (
            patch.object(resolution, "BreakableCudaGraphBackend") as target,
            patch.object(resolution, "FullCudaGraphBackend") as draft,
        ):
            self.assertIs(
                resolution.resolve_decode_backend(cast(BaseCudaGraphRunner, runner)),
                target.return_value,
            )
            model.is_draft_worker = True
            self.assertIs(
                resolution.resolve_decode_backend(cast(BaseCudaGraphRunner, runner)),
                draft.return_value,
            )
            self.assertEqual(config.decode.backend, "breakable")
            self.assertIs(model.server_args, args)

    def test_actual_spec_info_uses_admitted_dsa_causal_mask(self) -> None:
        from sglang.srt.layers.attention.dsa import sm80_dflash_verify as recipe

        backend = attention.DeepseekSparseAttnBackend.__new__(
            attention.DeepseekSparseAttnBackend
        )
        backend.device = torch.device("cuda")
        mask = torch.ones(64, dtype=torch.bool)
        owner = SimpleNamespace(
            model_runner=SimpleNamespace(
                spec_algorithm=SpeculativeAlgorithm.DFLASH,
                is_draft_worker=False,
                attn_backend=SimpleNamespace(full_attn_backend=backend),
            ),
            buffers=SimpleNamespace(custom_mask=mask),
            captured_req_width=8,
            ragged_verify_mode=False,
        )
        owner._capture_ragged_verify_layout = MethodType(
            decode.DecodeCudaGraphRunner._capture_ragged_verify_layout, owner
        )
        with (
            patch.object(recipe, "sm80_dflash_any_config_admitted", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)),
        ):
            spec = decode.DecodeCudaGraphRunner.get_spec_info(
                cast(decode.DecodeCudaGraphRunner, owner), 8
            )
        self.assertIsNone(spec.custom_mask)
        self.assertEqual(spec.draft_token_num, 8)
        self.assertIsNone(spec.ragged_verify_layout)
        with (
            patch.object(recipe, "sm80_dflash_any_config_admitted", return_value=False),
            patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)),
        ):
            spec = decode.DecodeCudaGraphRunner.get_spec_info(
                cast(decode.DecodeCudaGraphRunner, owner), 8
            )
        self.assertIs(spec.custom_mask, mask)

    def test_mask_policy_preserves_foreign_and_non_sm80_backends(self) -> None:
        from sglang.srt.layers.attention.dsa import sm80_dflash_verify as recipe

        backend = attention.DeepseekSparseAttnBackend.__new__(
            attention.DeepseekSparseAttnBackend
        )
        backend.device = torch.device("cuda")
        with (
            patch.object(recipe, "sm80_dflash_any_config_admitted", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
        ):
            self.assertEqual(
                dflash_utils.resolve_dflash_verify_mask_policy(backend),
                ("DeepseekSparseAttnBackend", True),
            )
        foreign = type("DeepseekSparseAttnBackend", (), {})()
        self.assertEqual(
            dflash_utils.resolve_dflash_verify_mask_policy(foreign),
            ("DeepseekSparseAttnBackend", True),
        )
        original_skip = type("TritonAttnBackend", (), {})()
        self.assertEqual(
            dflash_utils.resolve_dflash_verify_mask_policy(original_skip),
            ("TritonAttnBackend", False),
        )

    def test_mask_policy_uses_actual_common_recipe_refusals(self) -> None:
        from sglang.srt.layers.attention.dsa import sm80_dflash_verify as recipe

        backend = attention.DeepseekSparseAttnBackend.__new__(
            attention.DeepseekSparseAttnBackend
        )
        backend.device = torch.device("cuda")
        spec = SimpleNamespace(
            speculative_algorithm="DFLASH",
            speculative_dflash_implementation="ours",
            speculative_num_draft_tokens=8,
            speculative_dflash_block_size=8,
        )
        parallel = SimpleNamespace(attn_cp_size=1, attn_dcp_size=1)
        model = SimpleNamespace(kv_cache_dtype="bfloat16")
        kernel = SimpleNamespace(
            dsa_prefill_backend="tilelang",
            dsa_decode_backend="triton-sm80",
            dsa_paged_mqa_logits_backend="triton-sm80",
            dsa_ragged_mqa_logits_backend="torch-sm80",
            dsa_topk_backend="torch",
        )
        with (
            patch.object(recipe, "get_spec", return_value=spec),
            patch.object(recipe, "get_parallel", return_value=parallel),
            patch.object(recipe, "get_model", return_value=model),
            patch.object(
                recipe, "get_exec", return_value=SimpleNamespace(kernel=kernel)
            ),
            patch.object(
                recipe,
                "get_memory",
                return_value=SimpleNamespace(enable_hisparse=False),
            ),
            patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)),
        ):
            self.assertFalse(dflash_utils.resolve_dflash_verify_mask_policy(backend)[1])
            for owner, field, bad in (
                (spec, "speculative_algorithm", "EAGLE"),
                (spec, "speculative_dflash_implementation", "native"),
                (spec, "speculative_num_draft_tokens", 4),
                (parallel, "attn_cp_size", 2),
                (model, "kv_cache_dtype", "fp8_e4m3"),
                (kernel, "dsa_decode_backend", "flashmla_kv"),
            ):
                original = getattr(owner, field)
                with self.subTest(field=field):
                    setattr(owner, field, bad)
                    self.assertTrue(
                        dflash_utils.resolve_dflash_verify_mask_policy(backend)[1]
                    )
                    setattr(owner, field, original)

    def test_capture_stub_does_not_access_or_write_cache(self) -> None:
        config = SimpleNamespace(
            decode=SimpleNamespace(backend="breakable", bs=[1]),
            prefill=SimpleNamespace(backend="disabled"),
        )
        layer = SimpleNamespace(
            tp_q_head_num=2, head_dim=576, v_head_dim=512, is_cross_attention=False
        )
        q = torch.ones(8, 2, 512, dtype=torch.bfloat16)
        layout = SimpleNamespace(batch_size=1, query_rows=8)
        with (
            patch.object(attention, "require_sm80_dflash_verify", return_value=layout),
            patch.object(
                attention,
                "get_exec",
                return_value=SimpleNamespace(
                    graph=SimpleNamespace(cuda_graph_config=config)
                ),
            ),
            patch.object(attention.capture_mode, "is_capture_mode", True),
        ):
            # A bare object has no producer/cache attributes: capture must not use them.
            output = attention.DeepseekSparseAttnBackend._forward_sm80_dflash_verify(
                object(), q, None, None, layer, object(), True, None, None, None
            )
            self.assertEqual(output.shape, (8, 2, 512))
            self.assertEqual(output.dtype, torch.bfloat16)
            config.decode.backend = "full"
            with self.assertRaises(ValueError):
                attention._sm80_dflash_capture_stub(
                    cast(attention.DeepseekSparseAttnBackend, object()),
                    q,
                    None,
                    None,
                    layer,
                    object(),
                    True,
                    None,
                    None,
                    None,
                )


if __name__ == "__main__":
    unittest.main()
