import pytest
import sgl_kernel
import torch


def _b200_shadowkv_available() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (10, 0)
        and sgl_kernel.shadowkv_kernels_available()
    )


pytestmark = pytest.mark.skipif(
    not _b200_shadowkv_available(),
    reason="optional ShadowKV kernels require an enabled B200 wheel",
)


def _reconstruct_reference(u, sv, positions, inverse_frequencies):
    # Match the readable runtime boundary: BF16 inputs, FP32 accumulation in
    # the matrix product, and a BF16 result before FP32 RoPE arithmetic.
    reconstructed = torch.einsum("hnr,hrd->hnd", u[positions], sv).float()
    angles = positions.float().unsqueeze(-1) * inverse_frequencies
    cosine = torch.cat((angles.cos(), angles.cos()), dim=-1)
    sine = torch.cat((angles.sin(), angles.sin()), dim=-1)
    rotated_half = torch.cat(
        (-reconstructed[..., 32:], reconstructed[..., :32]), dim=-1
    )
    return (reconstructed * cosine + rotated_half * sine).to(torch.bfloat16)


def _plan_reference(arguments, max_reuse, chunk_size):
    (
        previous,
        previous_lengths,
        current,
        current_lengths,
        exact,
        exact_lengths,
        cached,
        current_generation,
    ) = arguments
    rows, current_width = current.shape
    plan = torch.full((rows, current_width, 3), -1, dtype=torch.int64, device="cuda")
    deduplicated = torch.full_like(exact, -1)
    counts = torch.zeros((rows, 3), dtype=torch.int32, device="cuda")
    for row in range(rows):
        previous_row = previous[row, : previous_lengths[row]].cpu().tolist()
        current_row = current[row, : current_lengths[row]].cpu().tolist()
        exact_row = exact[row, : exact_lengths[row]].cpu().tolist()
        exact_unique = list(dict.fromkeys(exact_row))
        if exact_unique:
            deduplicated[row, : len(exact_unique)] = torch.tensor(
                exact_unique, dtype=torch.int64, device="cuda"
            )
        reusable = (
            set(previous_row[:max_reuse])
            if cached[row].item() == current_generation[row].item()
            else set()
        )
        seen = set()
        hit_count = 0
        miss_count = 0
        for index, chunk in enumerate(current_row):
            kind = 0
            offset = -1
            if chunk not in seen and chunk not in exact_unique:
                if chunk in reusable:
                    kind = 1
                    hit_count += 1
                else:
                    kind = 2
                    offset = miss_count * chunk_size
                    miss_count += 1
            plan[row, index] = torch.tensor((kind, chunk, offset), device="cuda")
            seen.add(chunk)
        counts[row] = torch.tensor(
            (hit_count, miss_count, len(exact_unique)),
            dtype=torch.int32,
            device="cuda",
        )
    return plan, deduplicated, counts


@pytest.mark.parametrize(
    "tokens,heads,position_values",
    [
        (1, 1, []),
        (1, 1, [0]),
        (257, 3, [0, 1, 7, 8, 127, 128, 255, 256]),
        (4097, 8, [4096, 0, 4096, 31, 32, 2047, 2048]),
    ],
)
def test_shadowkv_reconstruct_rope_matches_fp32_reference(
    tokens, heads, position_values
):
    torch.manual_seed(20260822 + tokens + heads)
    u = (torch.randn((tokens, 160), device="cuda") * 0.125).to(torch.bfloat16)
    sv = (torch.randn((heads, 160, 64), device="cuda") * 0.125).to(torch.bfloat16)
    positions = torch.tensor(
        [position_values for _ in range(heads)], dtype=torch.int64, device="cuda"
    ).reshape(heads, -1)
    inverse = 1.0 / (
        500_000.0 ** (torch.arange(0, 64, 2, device="cuda", dtype=torch.float32) / 64)
    )
    expected = _reconstruct_reference(u, sv, positions, inverse)

    guard = 113
    output_elements = heads * len(position_values) * 64
    storage = torch.full(
        (guard + output_elements + guard,),
        91,
        dtype=torch.bfloat16,
        device="cuda",
    )
    output = storage[guard : guard + output_elements].view(
        heads, len(position_values), 64
    )
    actual = sgl_kernel.shadowkv_reconstruct_rope(u, sv, positions, inverse, out=output)

    torch.testing.assert_close(actual, expected, rtol=4e-2, atol=4e-2)
    assert torch.equal(actual, expected)
    assert torch.equal(storage[:guard], torch.full_like(storage[:guard], 91))
    assert torch.equal(storage[-guard:], torch.full_like(storage[-guard:], 91))
    repeated = sgl_kernel.shadowkv_reconstruct_rope(u, sv, positions, inverse)
    assert torch.equal(actual, repeated)


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("u-dtype", "u must use"),
        ("u-rank", r"\[tokens, 160\]"),
        ("sv-head-dim", r"\[kv_heads, 160, 64\]"),
        ("negative-position", "positions exceed"),
        ("large-position", "positions exceed"),
    ],
)
def test_shadowkv_reconstruct_rope_rejects_unsupported_input(mutation, message):
    u = torch.zeros((8, 160), dtype=torch.bfloat16, device="cuda")
    sv = torch.zeros((2, 160, 64), dtype=torch.bfloat16, device="cuda")
    positions = torch.zeros((2, 1), dtype=torch.int64, device="cuda")
    inverse = torch.ones((32,), dtype=torch.float32, device="cuda")
    if mutation == "u-dtype":
        u = u.float()
    elif mutation == "u-rank":
        u = u[:, :159].contiguous()
    elif mutation == "sv-head-dim":
        sv = sv[..., :63].contiguous()
    elif mutation == "negative-position":
        positions[0, 0] = -1
    elif mutation == "large-position":
        positions[0, 0] = 8

    with pytest.raises(ValueError, match=message):
        sgl_kernel.shadowkv_reconstruct_rope(u, sv, positions, inverse)


@pytest.mark.parametrize(
    "previous,previous_lengths,current,current_lengths,exact,exact_lengths,cached,current_generation,max_reuse",
    [
        ([[1, 4, 8, -1]], [3], [[8, 5, 4, 5]], [4], [[5, 5]], [2], [9], [9], 3),
        ([[1, 4, 8]], [3], [[8, 4, 1]], [3], [[10]], [1], [9], [10], 3),
        ([[1, 4, 8]], [3], [[8, 4, 1]], [3], [[10]], [1], [9], [9], 3),
        ([[]], [0], [[]], [0], [[]], [0], [1], [1], 0),
    ],
)
def test_shadowkv_plan_reuse_matches_readable_contract(
    previous,
    previous_lengths,
    current,
    current_lengths,
    exact,
    exact_lengths,
    cached,
    current_generation,
    max_reuse,
):
    def tensor(values, dtype):
        return torch.tensor(values, dtype=dtype, device="cuda")

    arguments = (
        tensor(previous, torch.int64),
        tensor(previous_lengths, torch.int32),
        tensor(current, torch.int64),
        tensor(current_lengths, torch.int32),
        tensor(exact, torch.int64),
        tensor(exact_lengths, torch.int32),
        tensor(cached, torch.int64),
        tensor(current_generation, torch.int64),
    )
    expected = _plan_reference(arguments, max_reuse, 8)
    actual = sgl_kernel.shadowkv_plan_reuse(
        *arguments, max_reuse_chunks=max_reuse, chunk_size=8
    )

    assert torch.equal(actual.plan, expected[0])
    assert torch.equal(actual.deduplicated_exact_chunks, expected[1])
    assert torch.equal(actual.counts, expected[2])
    repeated = sgl_kernel.shadowkv_plan_reuse(
        *arguments, max_reuse_chunks=max_reuse, chunk_size=8
    )
    assert torch.equal(actual.plan, repeated.plan)


def test_shadowkv_plan_reuse_rejects_invalid_active_region():
    previous = torch.tensor([[1, -1]], dtype=torch.int64, device="cuda")
    current = torch.tensor([[2]], dtype=torch.int64, device="cuda")
    exact = torch.tensor([[3]], dtype=torch.int64, device="cuda")
    with pytest.raises(ValueError, match="code 2"):
        sgl_kernel.shadowkv_plan_reuse(
            previous,
            torch.tensor([2], dtype=torch.int32, device="cuda"),
            current,
            torch.tensor([1], dtype=torch.int32, device="cuda"),
            exact,
            torch.tensor([1], dtype=torch.int32, device="cuda"),
            torch.tensor([1], dtype=torch.int64, device="cuda"),
            torch.tensor([1], dtype=torch.int64, device="cuda"),
            max_reuse_chunks=2,
            chunk_size=8,
        )
