import math

import pytest
import sgl_kernel
import torch
from sgl_kernel import shadowkv


def _supported_device_available() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() in {(8, 0), (10, 0)}
        and sgl_kernel.shadowkv_kernels_available()
    )


pytestmark = pytest.mark.skipif(
    not _supported_device_available(),
    reason="optional ShadowKV kernels require an enabled SM80 or SM100a wheel",
)


@pytest.mark.parametrize("capability", [(8, 0), (10, 0)])
def test_device_contract_accepts_supported_architectures(monkeypatch, capability):
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: capability
    )
    shadowkv._require_supported_device(torch.device("cuda"))


def test_device_contract_rejects_sm90(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    with pytest.raises(RuntimeError, match="8.0 or 10.0"):
        shadowkv._require_supported_device(torch.device("cuda"))


@pytest.mark.parametrize("rank", [64, 160, 256])
def test_reconstruct_matches_fp32_reference(rank):
    torch.manual_seed(rank)
    u = torch.randn((31, rank), device="cuda", dtype=torch.bfloat16)
    sv = torch.randn((2, rank, 128), device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([[0, 3, 30], [1, 5, 29]], device="cuda")
    actual = shadowkv.shadowkv_reconstruct(u, sv, positions)
    expected = torch.einsum("hnr,hrd->hnd", u[positions].float(), sv.float()).to(
        torch.bfloat16
    )
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.2)


def test_reconstruct_rope_matches_reference():
    torch.manual_seed(1)
    u = torch.randn((29, 160), device="cuda", dtype=torch.bfloat16)
    sv = torch.randn((2, 160, 64), device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([[0, 7, 28], [3, 11, 21]], device="cuda")
    inverse = 1.0 / (
        10_000 ** (torch.arange(0, 64, 2, device="cuda", dtype=torch.float32) / 64)
    )
    actual = shadowkv.shadowkv_reconstruct_rope(u, sv, positions, inverse)
    reconstructed = torch.einsum("hnr,hrd->hnd", u[positions].float(), sv.float())
    angles = positions.float().unsqueeze(-1) * inverse
    cosine = torch.cat((angles.cos(), angles.cos()), dim=-1)
    sine = torch.cat((angles.sin(), angles.sin()), dim=-1)
    rotated = torch.cat((-reconstructed[..., 32:], reconstructed[..., :32]), dim=-1)
    expected = (reconstructed * cosine + rotated * sine).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.2)


@pytest.mark.parametrize("head_dim", [64, 128])
def test_packed_gqa_matches_reference(head_dim):
    torch.manual_seed(head_dim)
    query = torch.randn((2, 4, head_dim), device="cuda", dtype=torch.bfloat16)
    keys = torch.randn((2, 2, 7, head_dim), device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    lengths = torch.tensor([7, 4], device="cuda", dtype=torch.int32)
    actual = shadowkv.shadowkv_packed_gqa(query, keys, values, lengths)
    expected = torch.empty_like(query)
    scale = 1.0 / math.sqrt(head_dim)
    for batch, length in enumerate(lengths.tolist()):
        for query_head in range(query.shape[1]):
            kv_head = query_head // (query.shape[1] // keys.shape[1])
            weights = torch.softmax(
                query[batch, query_head].float()
                @ keys[batch, kv_head, :length].float().T
                * scale,
                dim=-1,
            )
            expected[batch, query_head] = (
                weights @ values[batch, kv_head, :length].float()
            ).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.05)


def test_plan_reuse_classifies_hits_and_exact_chunks():
    previous = torch.tensor([[10, 20, -1]], device="cuda")
    previous_lengths = torch.tensor([2], device="cuda", dtype=torch.int32)
    current = torch.tensor([[20, 30, 20]], device="cuda")
    current_lengths = torch.tensor([3], device="cuda", dtype=torch.int32)
    exact = torch.tensor([[30, -1]], device="cuda")
    exact_lengths = torch.tensor([1], device="cuda", dtype=torch.int32)
    cached_generation = torch.tensor([5], device="cuda")
    current_generation = torch.tensor([5], device="cuda")

    result = shadowkv.shadowkv_plan_reuse(
        previous,
        previous_lengths,
        current,
        current_lengths,
        exact,
        exact_lengths,
        cached_generation,
        current_generation,
        max_reuse_chunks=2,
        chunk_size=8,
    )

    assert result.plan.cpu().tolist() == [[[1, 20, -1], [0, 30, -1], [0, 20, -1]]]
    assert result.deduplicated_exact_chunks.cpu().tolist() == [[30, -1]]
    assert result.counts.cpu().tolist() == [[1, 0, 1]]
