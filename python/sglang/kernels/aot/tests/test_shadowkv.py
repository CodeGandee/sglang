import pytest
import sgl_kernel
import sgl_kernel.shadowkv as shadowkv_module
import torch


def _supported_shadowkv_device_available() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() in {(8, 0), (10, 0)}
        and sgl_kernel.shadowkv_kernels_available()
    )


pytestmark = pytest.mark.skipif(
    not _supported_shadowkv_device_available(),
    reason="optional ShadowKV kernels require an enabled SM80 or SM100a wheel",
)


@pytest.mark.parametrize("capability", [(8, 0), (10, 0)])
def test_operation_device_contract_accepts_sm80_and_sm100a(monkeypatch, capability):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda device=None: capability,
    )
    shadowkv_module._require_supported_device(torch.device("cuda"))


def test_operation_device_contract_rejects_sm90_before_launch(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda device=None: (9, 0),
    )
    with pytest.raises(RuntimeError, match="8.0 or 10.0"):
        shadowkv_module._require_supported_device(torch.device("cuda"))


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


def _pre_rope_reconstruct_reference(u, sv, positions):
    return torch.einsum("hnr,hrd->hnd", u[positions], sv)


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


def _packed_gqa_reference(query, keys, values, lengths):
    outputs = []
    for row, length in enumerate(lengths.cpu().tolist()):
        if length == 0:
            outputs.append(torch.zeros_like(query[row]))
            continue
        groups = query.shape[1] // keys.shape[1]
        head_dim = query.shape[-1]
        grouped = query[row].float().reshape(keys.shape[1], groups, head_dim)
        scores = torch.einsum("hgd,hkd->hgk", grouped, keys[row, :, :length].float())
        weights = torch.softmax(scores * (head_dim**-0.5), dim=-1)
        output = torch.einsum("hgk,hkd->hgd", weights, values[row, :, :length].float())
        outputs.append(output.reshape(query.shape[1], head_dim).to(torch.bfloat16))
    return torch.stack(outputs)


@pytest.mark.parametrize(
    "batch,query_heads,kv_heads,head_dim,maximum_tokens,length_values",
    [
        (1, 4, 1, 64, 1, [1]),
        (2, 8, 2, 64, 17, [0, 13]),
        (3, 32, 8, 64, 257, [257, 31, 129]),
        (1, 32, 8, 64, 2465, [2465]),
        (1, 32, 8, 128, 1, [1]),
        (2, 32, 8, 128, 257, [257, 129]),
        (1, 32, 4, 128, 2049, [2049]),
    ],
)
def test_shadowkv_packed_gqa_matches_ragged_reference(
    batch, query_heads, kv_heads, head_dim, maximum_tokens, length_values
):
    torch.manual_seed(20260822 + batch + maximum_tokens)
    query = (torch.randn((batch, query_heads, head_dim), device="cuda") * 0.125).to(
        torch.bfloat16
    )
    keys = (
        torch.randn((batch, kv_heads, maximum_tokens, head_dim), device="cuda") * 0.125
    ).to(torch.bfloat16)
    values = (
        torch.randn((batch, kv_heads, maximum_tokens, head_dim), device="cuda") * 0.125
    ).to(torch.bfloat16)
    lengths = torch.tensor(length_values, dtype=torch.int32, device="cuda")
    expected = _packed_gqa_reference(query, keys, values, lengths)
    scratch = torch.empty(
        (batch, query_heads, maximum_tokens), dtype=torch.float32, device="cuda"
    )
    output = torch.empty_like(query)

    actual = sgl_kernel.shadowkv_packed_gqa(
        query, keys, values, lengths, weights=scratch, out=output
    )

    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    if head_dim == 64 and maximum_tokens == 2465:
        grouped = (
            query[0]
            .float()
            .reshape(
                kv_heads,
                query_heads // kv_heads,
                head_dim,
            )
        )
        expected_scores = torch.einsum("hgd,hkd->hgk", grouped, keys[0].float())
        expected_weights = torch.softmax(expected_scores * (head_dim**-0.5), dim=-1)
        assert torch.equal(
            scratch[0], expected_weights.reshape(query_heads, maximum_tokens)
        )


@pytest.mark.parametrize("head_dim", [64, 128])
def test_shadowkv_packed_gqa_replays_with_mutable_static_inputs(head_dim):
    query = torch.zeros((1, 4, head_dim), dtype=torch.bfloat16, device="cuda")
    keys = torch.zeros((1, 1, 17, head_dim), dtype=torch.bfloat16, device="cuda")
    values = torch.zeros_like(keys)
    lengths = torch.ones((1,), dtype=torch.int32, device="cuda")
    scratch = torch.empty((1, 4, 17), dtype=torch.float32, device="cuda")
    output = torch.empty_like(query)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            sgl_kernel.shadowkv_packed_gqa(
                query,
                keys,
                values,
                lengths,
                weights=scratch,
                out=output,
                validate_lengths=False,
            )
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        sgl_kernel.shadowkv_packed_gqa(
            query,
            keys,
            values,
            lengths,
            weights=scratch,
            out=output,
            validate_lengths=False,
        )

    torch.manual_seed(20260822)
    query.copy_(torch.randn_like(query))
    keys.copy_(torch.randn_like(keys))
    values.copy_(torch.randn_like(values))
    lengths.fill_(13)
    expected = _packed_gqa_reference(query, keys, values, lengths)
    graph.replay()

    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


def test_shadowkv_packed_gqa_rejects_unqualified_head_dimension():
    query = torch.zeros((1, 4, 96), dtype=torch.bfloat16, device="cuda")
    keys = torch.zeros((1, 1, 17, 96), dtype=torch.bfloat16, device="cuda")
    values = torch.zeros_like(keys)
    lengths = torch.ones((1,), dtype=torch.int32, device="cuda")

    with pytest.raises(ValueError, match="64 or 128"):
        sgl_kernel.shadowkv_packed_gqa(query, keys, values, lengths)


def test_shadowkv_packed_gqa_rejects_zero_kv_heads_before_division():
    query = torch.zeros((1, 4, 128), dtype=torch.bfloat16, device="cuda")
    keys = torch.zeros((1, 0, 17, 128), dtype=torch.bfloat16, device="cuda")
    values = torch.zeros_like(keys)
    lengths = torch.ones((1,), dtype=torch.int32, device="cuda")

    with pytest.raises(ValueError, match="positive"):
        sgl_kernel.shadowkv_packed_gqa(query, keys, values, lengths)


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
    strided_storage = torch.empty(
        (heads, len(position_values) + 3, 64),
        dtype=torch.bfloat16,
        device="cuda",
    )
    strided_output = strided_storage[:, : len(position_values)]
    sgl_kernel.shadowkv_reconstruct_rope(
        u,
        sv,
        positions,
        inverse,
        out=strided_output,
    )
    assert torch.equal(actual, strided_output)


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
    "rank,tokens,position_rows",
    [
        (64, 1, [[]]),
        (128, 1, [[0]]),
        (160, 257, [[0, 1, 7, 8, 127, 128, 256], [256, 8, 8, 1, 0, 128, 7]]),
        (
            256,
            4097,
            [
                [4096, 0, 4096, 31, 32, 2047, 2048],
                [2048, 2047, 32, 31, 4096, 0, 1],
                [1, 1, 2, 3, 5, 8, 13],
            ],
        ),
    ],
)
def test_shadowkv_reconstruct_matches_ragged_adversarial_reference(
    rank, tokens, position_rows
):
    torch.manual_seed(20260824 + rank + tokens)
    heads = len(position_rows)
    u = (torch.randn((tokens, rank), device="cuda") * 0.125).to(torch.bfloat16)
    sv = (torch.randn((heads, rank, 128), device="cuda") * 0.125).to(torch.bfloat16)
    positions = torch.tensor(position_rows, dtype=torch.int64, device="cuda").reshape(
        heads, -1
    )
    expected = _pre_rope_reconstruct_reference(u, sv, positions)

    guard = 117
    output_elements = heads * positions.shape[1] * 128
    storage = torch.full(
        (guard + output_elements + guard,),
        91,
        dtype=torch.bfloat16,
        device="cuda",
    )
    output = storage[guard : guard + output_elements].view(
        heads, positions.shape[1], 128
    )
    actual = sgl_kernel.shadowkv_reconstruct(u, sv, positions, out=output)

    torch.testing.assert_close(actual, expected, rtol=4e-2, atol=4e-2)
    assert torch.equal(actual, expected)
    assert torch.equal(storage[:guard], torch.full_like(storage[:guard], 91))
    assert torch.equal(storage[-guard:], torch.full_like(storage[-guard:], 91))
    repeated = sgl_kernel.shadowkv_reconstruct(u, sv, positions)
    assert torch.equal(actual, repeated)
    strided_storage = torch.empty(
        (heads, positions.shape[1] + 3, 128),
        dtype=torch.bfloat16,
        device="cuda",
    )
    strided_output = strided_storage[:, : positions.shape[1]]
    sgl_kernel.shadowkv_reconstruct(u, sv, positions, out=strided_output)
    assert torch.equal(actual, strided_output)


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("u-dtype", "u must use"),
        ("rank-32", "rank must be one of"),
        ("rank-96", "rank must be one of"),
        ("rank-192", "rank must be one of"),
        ("sv-rank", r"\[kv_heads, rank, 128\]"),
        ("sv-head-dim", r"\[kv_heads, rank, 128\]"),
        ("negative-position", "positions exceed"),
        ("large-position", "positions exceed"),
    ],
)
def test_shadowkv_reconstruct_rejects_unsupported_input(mutation, message):
    rank = int(mutation.removeprefix("rank-")) if mutation.startswith("rank-") else 160
    u = torch.zeros((8, rank), dtype=torch.bfloat16, device="cuda")
    sv_rank = 128 if mutation == "sv-rank" else rank
    head_dim = 127 if mutation == "sv-head-dim" else 128
    sv = torch.zeros((2, sv_rank, head_dim), dtype=torch.bfloat16, device="cuda")
    positions = torch.zeros((2, 1), dtype=torch.int64, device="cuda")
    if mutation == "u-dtype":
        u = u.float()
    elif mutation == "negative-position":
        positions[0, 0] = -1
    elif mutation == "large-position":
        positions[0, 0] = 8

    with pytest.raises(ValueError, match=message):
        sgl_kernel.shadowkv_reconstruct(u, sv, positions)


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


def test_shape_and_budget_guards_run_before_operator_launch(monkeypatch):
    def unexpected_launch(*args, **kwargs):
        pytest.fail("an invalid input reached the compiled operator")

    for name in (
        "_launch_shadowkv_packed_gqa",
        "_launch_shadowkv_plan_reuse",
        "_launch_shadowkv_reconstruct",
        "_launch_shadowkv_reconstruct_rope",
    ):
        monkeypatch.setattr(shadowkv_module, name, unexpected_launch)

    query = torch.zeros((1, 4, 96), dtype=torch.bfloat16, device="cuda")
    keys = torch.zeros((1, 1, 17, 96), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="64 or 128"):
        sgl_kernel.shadowkv_packed_gqa(
            query,
            keys,
            torch.zeros_like(keys),
            torch.ones((1,), dtype=torch.int32, device="cuda"),
        )

    positions = torch.zeros((1, 1), dtype=torch.int64, device="cuda")
    with pytest.raises(ValueError, match=r"\[tokens, 160\]"):
        sgl_kernel.shadowkv_reconstruct_rope(
            torch.zeros((8, 159), dtype=torch.bfloat16, device="cuda"),
            torch.zeros((1, 160, 64), dtype=torch.bfloat16, device="cuda"),
            positions,
            torch.ones((32,), dtype=torch.float32, device="cuda"),
        )
    with pytest.raises(ValueError, match="rank must be one of"):
        sgl_kernel.shadowkv_reconstruct(
            torch.zeros((8, 96), dtype=torch.bfloat16, device="cuda"),
            torch.zeros((1, 96, 128), dtype=torch.bfloat16, device="cuda"),
            positions,
        )

    chunks = torch.zeros((1, 1), dtype=torch.int64, device="cuda")
    lengths = torch.ones((1,), dtype=torch.int32, device="cuda")
    generations = torch.ones((1,), dtype=torch.int64, device="cuda")
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        sgl_kernel.shadowkv_plan_reuse(
            chunks,
            lengths,
            chunks,
            lengths,
            chunks,
            lengths,
            generations,
            generations,
            max_reuse_chunks=1,
            chunk_size=0,
        )
