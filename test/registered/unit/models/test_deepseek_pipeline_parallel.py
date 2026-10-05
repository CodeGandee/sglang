"""Exercise stage reduction ownership through native CPU control flow.

Model construction, decoder/MLP reduction, postprocessing and PP transport use
the implementation under test. Weight/kernel and collective arithmetic are
private CPU leaves; these tests do not qualify CUDA or NCCL execution.
"""

import os
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.srt.distributed import communication_op, parallel_state
from sglang.srt.layers import communicator as comm
from sglang.srt.layers import linear, moe
from sglang.srt.layers.communicator_dsa_cp import DSACPLayerCommunicator
from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.models import deepseek_v2 as model
from sglang.srt.runtime_context import get_context, get_forward, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from transformers import PretrainedConfig

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Norm(torch.nn.Module):
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        super().__init__()
        self.plain_calls = 0
        self.fused_calls = 0

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None = None,
        _addition: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self.plain_calls += 1
        represented = hidden if residual is None else hidden + residual
        widened = represented.to(torch.float64)
        normalized = (
            widened / (widened.square().mean(-1, keepdim=True) + 1).sqrt()
        ).to(torch.bfloat16)
        return normalized if residual is None else (normalized, represented)

    def forward_with_allreduce_fusion(
        self, hidden: torch.Tensor, residual: torch.Tensor, *, use_attn_tp_group: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert not use_attn_tp_group
        self.fused_calls += 1
        output = self.forward(
            communication_op.tensor_model_parallel_all_reduce(hidden), residual
        )
        assert isinstance(output, tuple)
        return output


class _Attention(torch.nn.Module):
    def __init__(self, **_kwargs: object) -> None:
        super().__init__()
        self.topk: torch.Tensor | None = None

    def prepare_qkv_latent(self, hidden: torch.Tensor, _batch: object) -> torch.Tensor:
        return hidden

    def maybe_use_decode_attn_tp(self, _batch: object):
        return nullcontext()

    def forward(self, *, hidden_states: torch.Tensor, **_kwargs: object):
        return hidden_states if self.topk is None else (hidden_states, self.topk)


class _Experts(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.moe_runner_config = SimpleNamespace(inplace=True)
        self.quant_method = SimpleNamespace()
        self.partial = torch.empty(0)

    def forward(self, _hidden: torch.Tensor, _topk: object) -> torch.Tensor:
        return self.partial.clone()


class _MoE(model.DeepseekV2MoE):
    def __init__(self, **_kwargs: object) -> None:
        torch.nn.Module.__init__(self)
        self.experts = _Experts()
        self.tp_size = 4
        self.layer_id = 0
        self.is_nextn = False
        self._fuse_shared_experts_inside_sbo = False
        self._shared_expert_tp1 = False
        self.routed_scaling_factor = 1.0
        self.gate = Mock(return_value=None)
        self.topk = Mock(return_value=None)

    def _maybe_quant_moe_input_once(self, _hidden: torch.Tensor) -> None:
        return None

    def _forward_shared_experts(self, *_args: object, **_kwargs: object) -> None:
        return None

    def forward(self, hidden: torch.Tensor, *_args: object) -> torch.Tensor:
        return self.forward_normal(hidden)


class _Projection(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, None]:
        return hidden, None


class _Dense(model.DeepseekV2MLP):
    def __init__(self, **_kwargs: object) -> None:
        torch.nn.Module.__init__(self)
        self.tp_size = 4
        self.swiglu_limit = None
        self.use_fused_clamp_act_mul = False
        self.gate_up_proj = _Projection()
        self.act_fn = torch.nn.Identity()
        self.down_proj = linear.RowParallelLinear.__new__(linear.RowParallelLinear)
        torch.nn.Module.__init__(self.down_proj)
        self.down_proj.input_is_parallel = True
        self.down_proj.tp_size = 4
        self.down_proj.tp_rank = 0
        self.down_proj.skip_bias_add = False
        self.down_proj.bias = None
        self.down_proj.use_dp_attention_reduce = False
        self.down_proj.reduce_results = True
        self.down_proj.use_decode_attn_tp = False
        self.down_proj.quant_method = SimpleNamespace(apply=Mock())


def _config(num_layers: int, *, dense: bool = False) -> PretrainedConfig:
    return PretrainedConfig(
        num_hidden_layers=num_layers,
        hidden_size=12,
        vocab_size=16,
        pad_token_id=0,
        first_k_dense_replace=num_layers if dense else 0,
        n_routed_experts=4,
        moe_layer_freq=1,
        rope_parameters={"rope_theta": 10000, "rope_type": "default"},
        max_position_embeddings=32,
        num_attention_heads=4,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=2,
        kv_lora_rank=2,
        intermediate_size=24,
        hidden_act="silu",
        rms_norm_eps=1e-6,
    )


def _context() -> comm.CommunicateContext:
    return comm.CommunicateContext(
        process_group_sizes={
            comm.ScatterMode.SCATTERED: 1,
            comm.ScatterMode.TP_ATTN_FULL: 4,
            comm.ScatterMode.FULL: 4,
            comm.ScatterMode.MOE_FULL: 4,
        },
        attn_tp_rank=0,
        attn_tp_size=4,
        attn_dp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        tp_size=4,
        tp_rank=0,
    )


def _partials(rows: int) -> list[torch.Tensor]:
    # Small dyadic operands have an exactly represented independent sum.
    base = torch.arange(rows * 12, dtype=torch.float64).reshape(rows, 12) / 32
    return [(base + (rank + 1) / 8).to(torch.bfloat16) for rank in range(4)]


class _SumLeaf:
    def __init__(self, expected: torch.Tensor) -> None:
        self.expected = expected
        self.entries: list[torch.Tensor] = []
        self.outputs: list[torch.Tensor] = []
        self.flags: list[tuple[bool, bool]] = []
        self.group = parallel_state.GroupCoordinator.__new__(
            parallel_state.GroupCoordinator
        )
        self.group.world_size = self.group.local_size = 4
        self.group.device_group = object()

    def __call__(self, value: torch.Tensor, *, group: object) -> None:
        assert group is self.group.device_group
        self.entries.append(value.clone())
        self.flags.append(
            (get_forward().fuse_mlp_allreduce, get_forward().mlp_reduce_scatter)
        )
        value.copy_(self.expected)
        self.outputs.append(value)


def _transport(outputs: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    """Run native PP send/receive with in-memory distributed numerical leaves."""
    packets: list[list[torch.Tensor]] = [[] for _ in outputs]
    metadata: list[list[tuple[str, object]]] = [[] for _ in outputs]
    for rank, output in enumerate(outputs):
        sender = SimpleNamespace(
            world_size=2,
            rank_in_group=0,
            ranks=(0, 1),
            device_group=object(),
            cpu_group=object(),
            send_object=lambda value, storage=metadata[rank], **_kwargs: storage.extend(
                value
            ),
        )
        tp_group = SimpleNamespace(world_size=4, rank_in_group=rank)
        with patch.object(
            torch.distributed,
            "send",
            side_effect=lambda value, *_args, storage=packets[rank], **_kwargs: (
                storage.append(value.clone())
            ),
        ):
            parallel_state.GroupCoordinator.send_tensor_dict(
                sender, output, all_gather_group=tp_group
            )

    def receive_rank(rank: int) -> dict[str, torch.Tensor]:
        leaf_index = 0

        def receive(value: torch.Tensor, **_kwargs: object) -> Mock:
            value.copy_(packets[rank][leaf_index])
            return Mock(wait=Mock())

        def gather(_value: torch.Tensor, *, dim: int) -> torch.Tensor:
            nonlocal leaf_index
            value = torch.cat([packet[leaf_index] for packet in packets], dim=dim)
            leaf_index += 1
            return value

        receiver = SimpleNamespace(
            world_size=2,
            rank_in_group=1,
            ranks=(0, 1),
            device_group=object(),
            cpu_group=object(),
            recv_object=lambda **_kwargs: metadata[rank],
        )
        tp_group = SimpleNamespace(world_size=4, rank_in_group=rank, all_gather=gather)
        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "irecv", side_effect=receive),
        ):
            result = parallel_state.GroupCoordinator.recv_tensor_dict(
                receiver, all_gather_group=tp_group
            )
        assert result is not None
        return result

    return [receive_rank(rank) for rank in range(4)]


class TestDeepseekPipelineParallel(CustomTestCase):
    """Stage boundaries transfer reduced values while local layers retain fusion."""

    def setUp(self) -> None:
        super().setUp()
        override = get_context().override_server_args(moe_dense_tp_size=4, dwdp_size=1)
        override.install()
        self.addCleanup(override.restore)
        self.enterContext(
            get_parallel().override(
                tp_size=4,
                moe_tp_size=4,
                moe_ep_size=1,
                attn_tp_size=4,
                attn_dp_size=1,
                attn_cp_size=1,
            )
        )
        self.enterContext(
            get_forward().scoped(attn_input_scattered=False, attn_inputs=None)
        )
        for owner, name, value in (
            (model, "DeepseekV2AttentionMLA", _Attention),
            (model, "DeepseekV2MoE", _MoE),
            (model, "DeepseekV2MLP", _Dense),
            (model, "RMSNorm", _Norm),
            (model, "_is_cuda", False),
            (model, "_is_musa", False),
            (model, "_is_gfx95_supported", False),
            (model, "_use_aiter_gfx95", False),
        ):
            self.enterContext(patch.object(owner, name, value))
        self.enterContext(
            patch.object(
                model, "VocabParallelEmbedding", return_value=torch.nn.Identity()
            )
        )
        self.enterContext(
            patch.object(model, "get_embedding_tp_kwargs", return_value={})
        )
        self.enterContext(
            patch.object(comm.CommunicateContext, "init_new", side_effect=_context)
        )
        spec = SimpleNamespace(speculative_algorithm=None)
        self.enterContext(patch.object(comm, "get_spec", return_value=spec))
        self.enterContext(patch.object(model, "get_spec", return_value=spec))
        self.enterContext(
            patch.object(
                model,
                "get_exec",
                return_value=SimpleNamespace(moe=SimpleNamespace(enable_eplb=False)),
            )
        )
        for owner in (model, comm, moe_utils):
            self.enterContext(
                patch.object(
                    owner, "get_moe_a2a_backend", return_value=moe.MoeA2ABackend.NONE
                )
            )
        for owner, name in (
            (comm, "is_enable_moe_cp_allgather"),
            (comm, "is_dp_attention_enabled"),
            (comm, "is_dsa_enable_prefill_cp"),
            (comm, "is_mla_prefill_cp_enabled"),
            (comm, "should_use_flashinfer_cutlass_moe_fp4_allgather"),
            (comm, "dsa_use_prefill_cp"),
            (comm, "mla_use_prefill_cp"),
            (model, "is_dsa_enable_prefill_cp"),
            (model, "is_prefill_context_parallel_enabled"),
            (model, "is_deepseek_dsa"),
            (moe_utils, "should_use_dp_reduce_scatterv"),
            (moe_utils, "should_use_flashinfer_cutlass_moe_fp4_allgather"),
        ):
            self.enterContext(patch.object(owner, name, return_value=False))
        self.enterContext(
            patch.object(comm, "apply_flashinfer_allreduce_fusion", return_value=True)
        )
        self.enterContext(patch.object(model, "maybe_prefetch_next_full_attention_kv"))
        self.enterContext(
            patch.object(
                linear,
                "use_symmetric_memory",
                side_effect=lambda *_args, **_kwargs: nullcontext(),
            )
        )

    def _model(self, num_layers: int, rank: int, size: int, *, dense: bool = False):
        pp_group = SimpleNamespace(
            rank_in_group=rank,
            world_size=size,
            is_first_rank=rank == 0,
            is_last_rank=rank == size - 1,
        )
        with patch.object(model, "get_pp_group", return_value=pp_group):
            return model.DeepseekV2Model(_config(num_layers, dense=dense))

    def test_actual_constructor_marks_only_actual_local_stage_end(self) -> None:
        cases = (
            (8, (8,), None),
            (8, (4, 4), None),
            (8, (2, 3, 3), None),
            (8, (1, 5, 2), "1,5,2"),
            (8, (0, 6, 2), "0,6,2"),
            (4, (1, 1, 1, 1), None),
        )
        for num_layers, counts, custom in cases:
            with patch.dict(os.environ):
                os.environ.pop("SGLANG_PP_LAYER_PARTITION", None)
                if custom is not None:
                    os.environ["SGLANG_PP_LAYER_PARTITION"] = custom
                start = 0
                for rank, count in enumerate(counts):
                    with self.subTest(counts=counts, rank=rank):
                        target = self._model(num_layers, rank, len(counts))
                        self.assertEqual(
                            (target.start_layer, target.end_layer),
                            (start, start + count),
                        )
                        for index in range(target.start_layer, target.end_layer):
                            communicator = target.layers[index].layer_communicator
                            self.assertTrue(hasattr(communicator, "is_pp_stage_end"))
                            self.assertEqual(
                                communicator.is_pp_stage_end,
                                index == target.end_layer - 1,
                            )
                            self.assertEqual(
                                communicator.is_last_layer, index == num_layers - 1
                            )
                            self.assertEqual(
                                communicator.layer_scatter_modes.layer_output_mode,
                                comm.ScatterMode.TP_ATTN_FULL,
                            )
                    start += count

    def test_actual_nextn_constructor_preserves_global_final_gate(self) -> None:
        decoder = model.DeepseekV2DecoderLayer(_config(8), 0, is_nextn=True)
        self.assertTrue(decoder.layer_communicator.is_last_layer)
        self.assertTrue(hasattr(decoder.layer_communicator, "is_pp_stage_end"))
        self.assertFalse(decoder.layer_communicator.is_pp_stage_end)
        batch = SimpleNamespace(input_ids=torch.tensor([1]))
        self.assertFalse(
            decoder.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(batch)
        )

    def test_cp_communicator_inherits_separate_stage_marker(self) -> None:
        modes = comm.LayerScatterModes(
            comm.ScatterMode.SCATTERED,
            comm.ScatterMode.SCATTERED,
            comm.ScatterMode.FULL,
            comm.ScatterMode.SCATTERED,
            comm.ScatterMode.SCATTERED,
        )
        communicator = DSACPLayerCommunicator(
            modes, _Norm(), _Norm(), allow_reduce_scatter=True
        )
        self.assertTrue(hasattr(communicator, "is_pp_stage_end"))
        self.assertFalse(communicator.is_pp_stage_end)
        communicator.is_pp_stage_end = True
        self.assertFalse(communicator.is_last_layer)
        self.assertFalse(
            communicator.should_fuse_mlp_allreduce_with_next_layer(
                SimpleNamespace(input_ids=torch.tensor([1]))
            )
        )

    def _run_layer(
        self,
        decoder: model.DeepseekV2DecoderLayer,
        partial: torch.Tensor,
        residual: torch.Tensor,
        topk: torch.Tensor,
    ):
        decoder.self_attn.topk = topk
        if isinstance(decoder.mlp, _MoE):
            decoder.mlp.experts.partial = partial
        else:
            decoder.mlp.down_proj.quant_method.apply.side_effect = (
                lambda *_args, **_kwargs: partial.clone()
            )
        batch = SimpleNamespace(
            input_ids=torch.ones(partial.shape[0], dtype=torch.int64)
        )
        with (
            patch.object(
                decoder.layer_communicator,
                "prepare_attn_and_capture_last_layer_outputs",
                side_effect=lambda hidden, residual, *_args, **_kwargs: (
                    hidden,
                    residual,
                ),
            ),
            patch.object(
                decoder.layer_communicator,
                "prepare_mlp",
                side_effect=lambda hidden, residual, *_args: (hidden, residual),
            ),
        ):
            return decoder(
                torch.arange(partial.shape[0]),
                torch.zeros_like(partial),
                batch,
                residual,
                None,
            )

    def test_stage_end_reduces_before_native_pp_transport(self) -> None:
        for dense in (False, True):
            for rows in (1, 4):
                with self.subTest(dense=dense, rows=rows):
                    partials = _partials(rows)
                    expected = sum(
                        (value.to(torch.float64) for value in partials),
                        torch.zeros_like(partials[0], dtype=torch.float64),
                    ).to(torch.bfloat16)
                    residual = torch.full_like(expected, 0.5)
                    topk = torch.arange(rows * 8, dtype=torch.int32).reshape(rows, 8)
                    reduction = _SumLeaf(expected)

                    outputs: list[dict[str, torch.Tensor]] = []
                    with (
                        patch.object(
                            communication_op,
                            "get_tp_group",
                            return_value=reduction.group,
                        ),
                        patch.object(
                            linear, "get_tp_group", return_value=reduction.group
                        ),
                        patch.object(
                            parallel_state, "is_shm_available", return_value=False
                        ),
                        patch.object(
                            torch.distributed, "all_reduce", side_effect=reduction
                        ),
                    ):
                        for partial in partials:
                            target = self._model(4, 0, 2, dense=dense)
                            decoder = target.layers[target.end_layer - 1]
                            self.assertIs(
                                decoder.layer_communicator._communicate_summable_tensor_pair_fn,
                                comm.CommunicateSummableTensorPairFn._trivial,
                            )
                            hidden, returned_residual, returned_topk = self._run_layer(
                                decoder, partial, residual, topk
                            )
                            self.assertIs(returned_residual, residual)
                            self.assertIs(returned_topk, topk)
                            outputs.append(
                                {
                                    "hidden_states": hidden,
                                    "residual": returned_residual,
                                    "topk_indices": returned_topk,
                                }
                            )
                        received = _transport(outputs)
                    for received_rank, values in enumerate(received):
                        with self.subTest(received_rank=received_rank):
                            self.assertTrue(
                                torch.equal(values["hidden_states"], expected)
                            )
                            self.assertTrue(torch.equal(values["residual"], residual))
                            self.assertTrue(torch.equal(values["topk_indices"], topk))
                            self.assertFalse(
                                hasattr(
                                    values["hidden_states"],
                                    "_sglang_needs_allreduce_fusion",
                                )
                            )
                            consumer_target = self._model(4, 1, 2, dense=dense)
                            consumer = consumer_target.layers[
                                consumer_target.start_layer
                            ]
                            normalized, represented = (
                                consumer.layer_communicator.prepare_attn(
                                    values["hidden_states"],
                                    values["residual"],
                                    SimpleNamespace(
                                        input_ids=torch.ones(rows, dtype=torch.int64)
                                    ),
                                )
                            )
                            self.assertEqual(consumer.input_layernorm.plain_calls, 1)
                            self.assertEqual(consumer.input_layernorm.fused_calls, 0)
                            self.assertTrue(
                                torch.equal(represented, expected + residual)
                            )
                            widened = (expected + residual).to(torch.float64)
                            independent_norm = (
                                widened
                                / (widened.square().mean(-1, keepdim=True) + 1).sqrt()
                            ).to(torch.bfloat16)
                            self.assertTrue(torch.equal(normalized, independent_norm))
                    self.assertEqual(len(reduction.entries), 4)
                    self.assertEqual(reduction.flags, [(False, False)] * 4)
                    for partial, entry, reduction_output, output in zip(
                        partials, reduction.entries, reduction.outputs, outputs
                    ):
                        self.assertTrue(torch.equal(entry, partial))
                        self.assertTrue(torch.equal(output["hidden_states"], expected))
                        self.assertIs(output["hidden_states"], reduction_output)
                        self.assertFalse(
                            hasattr(
                                output["hidden_states"],
                                "_sglang_needs_allreduce_fusion",
                            )
                        )

    def test_internal_layer_defers_and_next_local_layer_consumes_once(self) -> None:
        for dense in (False, True):
            for rows in (1, 4):
                with self.subTest(dense=dense, rows=rows):
                    target = self._model(4, 0, 2, dense=dense)
                    producer, consumer = target.layers[0], target.layers[1]
                    partials = _partials(rows)
                    expected = sum(
                        (value.to(torch.float64) for value in partials),
                        torch.zeros_like(partials[0], dtype=torch.float64),
                    ).to(torch.bfloat16)
                    residual = torch.full_like(expected, 0.5)
                    topk = torch.zeros((rows, 8), dtype=torch.int32)
                    reduction = _SumLeaf(expected)

                    with (
                        patch.object(
                            communication_op,
                            "get_tp_group",
                            return_value=reduction.group,
                        ),
                        patch.object(
                            linear, "get_tp_group", return_value=reduction.group
                        ),
                        patch.object(
                            parallel_state, "is_shm_available", return_value=False
                        ),
                        patch.object(
                            torch.distributed, "all_reduce", side_effect=reduction
                        ),
                    ):
                        hidden, returned_residual, _ = self._run_layer(
                            producer, partials[0], residual, topk
                        )
                        self.assertTrue(hidden._sglang_needs_allreduce_fusion)
                        self.assertTrue(torch.equal(hidden, partials[0]))
                        self.assertEqual(len(reduction.entries), 0)
                        batch = SimpleNamespace(
                            input_ids=torch.ones(rows, dtype=torch.int64)
                        )
                        normalized, represented = (
                            consumer.layer_communicator.prepare_attn(
                                hidden, returned_residual, batch
                            )
                        )
                    self.assertEqual(len(reduction.entries), 1)
                    self.assertEqual(consumer.input_layernorm.fused_calls, 1)
                    self.assertTrue(torch.equal(represented, expected + residual))
                    widened = (expected + residual).to(torch.float64)
                    independent_norm = (
                        widened / (widened.square().mean(-1, keepdim=True) + 1).sqrt()
                    ).to(torch.bfloat16)
                    self.assertTrue(torch.equal(normalized, independent_norm))
                    self.assertFalse(
                        hasattr(normalized, "_sglang_needs_allreduce_fusion")
                    )


if __name__ == "__main__":
    unittest.main()
