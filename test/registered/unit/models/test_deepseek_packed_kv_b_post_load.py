"""Check packed MLA post-load against independent scalar W8/group references."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.linear import ColumnParallelLinear
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (
    CompressedTensorsWNA16,
)
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _scalar_pack(rows):
    """Encode offset-binary bytes directly, independently of dependency helpers."""
    packed = []
    for row in rows:
        words = []
        for start in range(0, len(row), 4):
            word = sum((row[start + lane] + 128) << (8 * lane) for lane in range(4))
            words.append(word if word < 2**31 else word - 2**32)
        packed.append(words)
    return torch.tensor(packed, dtype=torch.int32)


def _inputs():
    # Each row covers every signed W8 code, with distinct non-dyadic group scales.
    values = [
        [((row * 37 + col) % 256) - 128 for col in range(256)] for row in range(16)
    ]
    scales = torch.tensor(
        [[0.1 * (row + 1), 0.37 * (row + 1)] for row in range(16)],
        dtype=torch.bfloat16,
    )
    reference = torch.tensor(
        [
            [
                float(value) * float(scales[row, col // 128])
                for col, value in enumerate(values[row])
            ]
            for row in range(16)
        ],
        dtype=torch.bfloat16,
    )
    return _scalar_pack(values), scales, reference


def _projection(packed, scales, *, tp_rank=0, tp_size=1):
    projection = ColumnParallelLinear.__new__(ColumnParallelLinear)
    torch.nn.Module.__init__(projection)
    rows, packed_cols = packed.shape
    width = packed_cols * 4
    projection.input_size = width
    projection.output_size = rows
    projection.output_size_per_partition = rows // tp_size
    projection.tp_rank = tp_rank
    projection.tp_size = tp_size
    projection.use_presharded_weights = False
    projection.scheme = CompressedTensorsWNA16(
        strategy="group", num_bits=8, group_size=128
    )
    projection.scheme.create_weights(
        projection,
        output_size=rows,
        input_size=width,
        output_partition_sizes=[rows // tp_size],
        input_size_per_partition=width,
        params_dtype=torch.bfloat16,
        weight_loader=projection.weight_loader,
    )
    projection.weight_loader(projection.weight_packed, packed)
    projection.weight_loader(projection.weight_scale, scales)
    projection.weight_loader(projection.weight_shape, torch.tensor([rows, width]))
    return projection


def _fixture(projection, *, initial_scale=1.0):
    attention = SimpleNamespace(
        kv_b_proj=projection,
        qk_nope_head_dim=2,
        v_head_dim=2,
        w_kc=None,
        w_vc=None,
        w_scale=initial_scale,
    )
    fixture = SimpleNamespace(
        model=SimpleNamespace(
            start_layer=0,
            end_layer=1,
            layers=[SimpleNamespace(self_attn=attention)],
        ),
        config=SimpleNamespace(
            num_hidden_layers=1, architectures=["GlmMoeDsaForCausalLM"]
        ),
        quant_config=CompressedTensorsConfig({}, [], "pack-quantized", {}, []),
    )
    return fixture, attention


class TestPackedKVBPostLoad(unittest.TestCase):
    def test_scalar_corner_encoding_is_offset_binary_low_byte_first(self):
        packed = _scalar_pack([[-128, -127, -1, 0, 1, 126, 127, -64]])
        self.assertEqual(
            [int(value) & 0xFFFFFFFF for value in packed[0]],
            [0x807F0100, 0x40FFFE81],
        )

    def test_actual_post_load_matches_independent_bf16_and_preserves_packed_projection(
        self,
    ):
        packed, scales, reference = _inputs()
        for tp_size in (1, 2):
            for tp_rank in range(tp_size):
                for initial_scale in (1.0, None):
                    with self.subTest(
                        tp_size=tp_size, tp_rank=tp_rank, initial_scale=initial_scale
                    ):
                        projection = _projection(
                            packed, scales, tp_rank=tp_rank, tp_size=tp_size
                        )
                        fixture, attention = _fixture(
                            projection, initial_scale=initial_scale
                        )
                        before = {
                            name: (
                                parameter,
                                parameter.data_ptr(),
                                parameter.detach().clone(),
                            )
                            for name, parameter in projection.named_parameters()
                        }
                        scheme = projection.scheme
                        DeepseekV2WeightLoaderMixin.post_load_weights(fixture)
                        rows = reference.shape[0] // tp_size
                        local = reference[tp_rank * rows : (tp_rank + 1) * rows]
                        heads = rows // 4
                        expected_k = local.reshape(heads, 4, 256)[:, :2]
                        expected_v = local.reshape(heads, 4, 256)[:, 2:].transpose(1, 2)
                        self.assertTrue(torch.equal(attention.w_kc, expected_k))
                        self.assertTrue(torch.equal(attention.w_vc, expected_v))
                        self.assertEqual(attention.w_kc.dtype, torch.bfloat16)
                        self.assertEqual(attention.w_vc.dtype, torch.bfloat16)
                        self.assertEqual(attention.w_scale, 1.0)
                        self.assertIs(projection.scheme, scheme)
                        self.assertFalse(hasattr(projection, "weight"))
                        self.assertEqual(
                            set(dict(projection.named_parameters())), set(before)
                        )
                        for name, (parameter, pointer, value) in before.items():
                            self.assertIs(getattr(projection, name), parameter)
                            self.assertEqual(parameter.data_ptr(), pointer)
                            self.assertTrue(torch.equal(parameter, value))
                        # Original descriptors remain global; native column loading
                        # has sliced packed/scales rows independently on each rank.
                        self.assertTrue(
                            torch.equal(
                                projection.weight_shape, torch.tensor([16, 256])
                            )
                        )
                        query = (
                            torch.arange(heads * 2, dtype=torch.float64).reshape(
                                heads, 1, 2
                            )
                            / 4
                        )
                        latent = (
                            torch.arange(heads * 256, dtype=torch.float64).reshape(
                                heads, 1, 256
                            )
                            / 128
                        )
                        self.assertTrue(
                            torch.equal(
                                torch.bmm(query, attention.w_kc.to(torch.float64)),
                                torch.bmm(query, expected_k.to(torch.float64)),
                            )
                        )
                        self.assertTrue(
                            torch.equal(
                                torch.bmm(latent, attention.w_vc.to(torch.float64)),
                                torch.bmm(latent, expected_v.to(torch.float64)),
                            )
                        )
                        # Match BF16 serving consumers separately from the FP64
                        # semantic mapping checks, including the V output stride.
                        bf16_query = (
                            torch.arange(heads * 3 * 2)
                            .reshape(heads, 3, 2)
                            .to(torch.bfloat16)
                            / 8
                        )
                        bf16_latent = (
                            torch.arange(heads * 3 * 256).reshape(heads, 3, 256) % 17
                        ).to(torch.bfloat16) / 8
                        self.assertTrue(
                            torch.equal(
                                torch.bmm(bf16_query, attention.w_kc),
                                torch.bmm(bf16_query, expected_k.contiguous()),
                            )
                        )
                        output = torch.empty((3, heads * 2), dtype=torch.bfloat16)
                        output_view = output.view(3, heads, 2).transpose(0, 1)
                        self.assertFalse(output_view.is_contiguous())
                        torch.bmm(bf16_latent, attention.w_vc, out=output_view)
                        expected_output = (
                            torch.bmm(bf16_latent, expected_v.contiguous())
                            .transpose(0, 1)
                            .reshape(3, heads * 2)
                        )
                        self.assertTrue(torch.equal(output, expected_output))

    def test_actual_ordinary_bf16_post_load_is_unchanged(self):
        _, _, reference = _inputs()
        projection = torch.nn.Module()
        projection.register_parameter(
            "weight", torch.nn.Parameter(reference, requires_grad=False)
        )
        fixture, attention = _fixture(projection)
        fixture.quant_config = None
        DeepseekV2WeightLoaderMixin.post_load_weights(fixture)
        self.assertTrue(
            torch.equal(attention.w_kc, reference.reshape(4, 4, 256)[:, :2])
        )
        self.assertTrue(
            torch.equal(
                attention.w_vc, reference.reshape(4, 4, 256)[:, 2:].transpose(1, 2)
            )
        )
        self.assertEqual(attention.w_scale, 1.0)

    def test_repacked_or_unsupported_formats_are_refused(self):
        packed, scales, _ = _inputs()
        mutations = (
            lambda p: setattr(
                p,
                "weight_packed",
                torch.nn.Parameter(
                    p.weight_packed.detach().clone(), requires_grad=False
                ),
            ),
            lambda p: setattr(p.scheme, "symmetric", False),
            lambda p: setattr(p.scheme, "has_g_idx", True),
            lambda p: setattr(p.scheme, "group_size", -1),
            lambda p: p.weight_shape.data.copy_(torch.tensor([16, 512])),
            lambda p: setattr(p.weight_scale, "data", p.weight_scale.detach()[:, :1]),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                projection = _projection(packed, scales)
                mutate(projection)
                fixture, attention = _fixture(projection)
                with self.assertRaises(ValueError):
                    DeepseekV2WeightLoaderMixin.post_load_weights(fixture)
                self.assertIsNone(attention.w_kc)
                self.assertIsNone(attention.w_vc)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
