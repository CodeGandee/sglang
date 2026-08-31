import math

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


def _device_plan_reference(arguments, plan_capacity):
    (
        selected,
        selected_lengths,
        exact,
        exact_lengths,
        temporal,
        validity,
        publications,
        temporal_request_generations,
        temporal_layout_generations,
        row_indices,
        row_generations,
        plan_slots,
    ) = arguments
    del publications
    rows, selected_capacity = selected.shape
    request_slots, local_layers, kv_heads, temporal_capacity = temporal.shape
    kinds = torch.full(
        (2, rows, selected_capacity), -1, dtype=torch.int8, device="cuda"
    )
    sources = torch.full_like(kinds, -1, dtype=torch.int32)
    destinations = torch.full_like(kinds, -1, dtype=torch.int32)
    misses = torch.full_like(kinds, -1, dtype=torch.int32)
    counts = torch.zeros((2, rows, 2), dtype=torch.int32, device="cuda")
    for row in range(rows):
        request_slot, local_layer, kv_head = row_indices[row].cpu().tolist()
        request_generation, layout_generation, _ = row_generations[row].cpu().tolist()
        selected_row = selected[row, : selected_lengths[row]].cpu().tolist()
        exact_row = exact[row, : exact_lengths[row]].cpu().tolist()
        temporal_row = temporal[request_slot, local_layer, kv_head].cpu().tolist()
        owner_matches = (
            temporal_request_generations[request_slot].item() == request_generation
            and temporal_layout_generations[request_slot].item() == layout_generation
        )
        source_by_chunk = (
            {chunk: index for index, chunk in enumerate(temporal_row) if chunk >= 0}
            if owner_matches
            else {}
        )
        seen = set()
        exact_set = set(exact_row)
        for selected_ordinal, chunk in enumerate(selected_row):
            if chunk in seen or chunk in exact_set:
                continue
            seen.add(chunk)
            for component in range(2):
                destinations[component, row, selected_ordinal] = (
                    (component * plan_capacity + plan_slots[row].item()) * kv_heads
                    + kv_head
                ) * selected_capacity + selected_ordinal
                temporal_ordinal = source_by_chunk.get(chunk)
                hit = (
                    temporal_ordinal is not None
                    and validity[
                        component,
                        request_slot,
                        local_layer,
                        kv_head,
                        temporal_ordinal,
                    ].item()
                    == 1
                )
                if hit:
                    kinds[component, row, selected_ordinal] = 1
                    sources[component, row, selected_ordinal] = (
                        (
                            (component * request_slots + request_slot) * local_layers
                            + local_layer
                        )
                        * kv_heads
                        + kv_head
                    ) * temporal_capacity + temporal_ordinal
                    counts[component, row, 0] += 1
                else:
                    kinds[component, row, selected_ordinal] = 2
                    misses[component, row, selected_ordinal] = counts[component, row, 1]
                    counts[component, row, 1] += 1
    return kinds, sources, destinations, misses, counts


def _device_plan_arguments():
    selected = torch.tensor([[8, 5, 8, 99, 6]], dtype=torch.int32, device="cuda")
    selected_lengths = torch.tensor([5], dtype=torch.int32, device="cuda")
    exact = torch.tensor([[99, 99, -1]], dtype=torch.int32, device="cuda")
    exact_lengths = torch.tensor([2], dtype=torch.int32, device="cuda")
    temporal = torch.full((1, 2, 2, 3), -1, dtype=torch.int32, device="cuda")
    temporal[0, 1, 1] = torch.tensor([5, 6, 8], dtype=torch.int32, device="cuda")
    validity = torch.zeros((2, 1, 2, 2, 3), dtype=torch.uint8, device="cuda")
    validity[0, 0, 1, 1] = torch.tensor([1, 0, 1], dtype=torch.uint8, device="cuda")
    validity[1, 0, 1, 1] = torch.tensor([0, 1, 1], dtype=torch.uint8, device="cuda")
    publications = torch.full((2, 1, 2, 2, 3), -1, dtype=torch.int64, device="cuda")
    publications[0, 0, 1, 1] = torch.tensor(
        [4, -1, 5], dtype=torch.int64, device="cuda"
    )
    publications[1, 0, 1, 1] = torch.tensor(
        [-1, 6, 5], dtype=torch.int64, device="cuda"
    )
    return (
        selected,
        selected_lengths,
        exact,
        exact_lengths,
        temporal,
        validity,
        publications,
        torch.tensor([7], dtype=torch.int64, device="cuda"),
        torch.tensor([3], dtype=torch.int64, device="cuda"),
        torch.tensor([[0, 1, 1]], dtype=torch.int32, device="cuda"),
        torch.tensor([[7, 3, 9]], dtype=torch.int64, device="cuda"),
        torch.tensor([1], dtype=torch.int32, device="cuda"),
    )


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
    if head_dim == 64:
        assert torch.equal(actual, expected)
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


def test_shadowkv_plan_device_matches_component_aware_readable_contract():
    arguments = _device_plan_arguments()
    expected = _device_plan_reference(arguments, plan_capacity=2)

    actual = sgl_kernel.shadowkv_plan_device(*arguments, plan_capacity=2)
    repeated = sgl_kernel.shadowkv_plan_device(*arguments, plan_capacity=2)

    assert not actual.error_codes.any().item()
    assert torch.equal(actual.component_kinds, expected[0])
    assert torch.equal(actual.source_slots, expected[1])
    assert torch.equal(actual.destination_slots, expected[2])
    assert torch.equal(actual.miss_ordinals, expected[3])
    assert torch.equal(actual.counts, expected[4])
    assert torch.equal(actual.component_kinds, repeated.component_kinds)
    assert actual.selected_chunk_ids.data_ptr() == arguments[0].data_ptr()
    assert actual.row_indices.data_ptr() == arguments[9].data_ptr()


def test_shadowkv_plan_device_matches_random_ragged_and_stale_owner_rows():
    generator = torch.Generator(device="cuda").manual_seed(20260831)
    rows = 24
    selected_capacity = 17
    exact_capacity = 7
    temporal_capacity = 11
    request_slots = 3
    local_layers = 2
    kv_heads = 4
    plan_capacity = 5
    selected = torch.randint(
        0,
        23,
        (rows, selected_capacity),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    selected_lengths = torch.randint(
        0,
        selected_capacity + 1,
        (rows,),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    exact = torch.randint(
        0,
        23,
        (rows, exact_capacity),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    exact_lengths = torch.randint(
        0,
        exact_capacity + 1,
        (rows,),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    temporal = (
        torch.arange(temporal_capacity, dtype=torch.int32, device="cuda")
        .expand(request_slots, local_layers, kv_heads, -1)
        .contiguous()
    )
    validity = torch.randint(
        0,
        2,
        (2, request_slots, local_layers, kv_heads, temporal_capacity),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    publications = torch.where(
        validity.bool(),
        torch.ones_like(validity, dtype=torch.int64),
        torch.full_like(validity, -1, dtype=torch.int64),
    )
    request_generations = torch.tensor([7, 8, 9], dtype=torch.int64, device="cuda")
    layout_generations = torch.tensor([3, 3, 4], dtype=torch.int64, device="cuda")
    row_indices = torch.empty((rows, 3), dtype=torch.int32, device="cuda")
    row_indices[:, 0] = torch.arange(rows, device="cuda") % request_slots
    row_indices[:, 1] = torch.arange(rows, device="cuda") % local_layers
    row_indices[:, 2] = torch.arange(rows, device="cuda") % kv_heads
    row_generations = torch.empty((rows, 3), dtype=torch.int64, device="cuda")
    row_generations[:, 0] = request_generations[row_indices[:, 0].long()]
    row_generations[:, 1] = layout_generations[row_indices[:, 0].long()]
    row_generations[:, 2] = 9
    row_generations[::3, 0] += 1
    plan_slots = torch.arange(rows, dtype=torch.int32, device="cuda") % plan_capacity
    arguments = (
        selected,
        selected_lengths,
        exact,
        exact_lengths,
        temporal,
        validity,
        publications,
        request_generations,
        layout_generations,
        row_indices,
        row_generations,
        plan_slots,
    )
    expected = _device_plan_reference(arguments, plan_capacity)

    actual = sgl_kernel.shadowkv_plan_device(*arguments, plan_capacity=plan_capacity)

    assert not actual.error_codes.any().item()
    assert torch.equal(actual.component_kinds, expected[0])
    assert torch.equal(actual.source_slots, expected[1])
    assert torch.equal(actual.destination_slots, expected[2])
    assert torch.equal(actual.miss_ordinals, expected[3])
    assert torch.equal(actual.counts, expected[4])


@pytest.mark.parametrize(
    "mutation,error_code",
    [
        ("invalid-length", 1),
        ("invalid-chunk", 2),
        ("invalid-identity", 3),
        ("invalid-plan-slot", 4),
        ("invalid-validity", 5),
        ("duplicate-temporal", 6),
        ("invalid-publication", 7),
    ],
)
def test_shadowkv_plan_device_defers_row_errors(mutation, error_code):
    arguments = list(_device_plan_arguments())
    if mutation == "invalid-length":
        arguments[1][0] = 6
    elif mutation == "invalid-chunk":
        arguments[0][0, 0] = -1
    elif mutation == "invalid-identity":
        arguments[9][0, 1] = 2
    elif mutation == "invalid-plan-slot":
        arguments[11][0] = 2
    elif mutation == "invalid-validity":
        arguments[5][0, 0, 1, 1, 0] = 2
    elif mutation == "duplicate-temporal":
        arguments[4][0, 1, 1, 1] = 5
    elif mutation == "invalid-publication":
        arguments[6][0, 0, 1, 1, 0] = 9

    plan = sgl_kernel.shadowkv_plan_device(*arguments, plan_capacity=2)

    assert plan.error_codes.cpu().tolist() == [error_code]
    assert torch.equal(plan.component_kinds, torch.full_like(plan.component_kinds, -1))
    assert torch.equal(plan.source_slots, torch.full_like(plan.source_slots, -1))
    assert torch.equal(
        plan.destination_slots, torch.full_like(plan.destination_slots, -1)
    )
    assert torch.equal(plan.miss_ordinals, torch.full_like(plan.miss_ordinals, -1))
    assert torch.equal(plan.counts, torch.zeros_like(plan.counts))


def test_shadowkv_plan_device_preserves_guard_regions():
    arguments = _device_plan_arguments()
    expected = _device_plan_reference(arguments, plan_capacity=2)
    rows, selected_capacity = arguments[0].shape
    guard = 31

    def guarded(shape, dtype, fill):
        elements = math.prod(shape)
        storage = torch.full(
            (guard + elements + guard,), fill, dtype=dtype, device="cuda"
        )
        return storage, storage[guard : guard + elements].view(shape)

    kind_storage, kinds = guarded((2, rows, selected_capacity), torch.int8, 99)
    source_storage, sources = guarded((2, rows, selected_capacity), torch.int32, 997)
    destination_storage, destinations = guarded(
        (2, rows, selected_capacity), torch.int32, 998
    )
    miss_storage, misses = guarded((2, rows, selected_capacity), torch.int32, 999)
    count_storage, counts = guarded((2, rows, 2), torch.int32, 1000)
    error_storage, errors = guarded((rows,), torch.int32, 1001)
    torch.ops.sgl_kernel.shadowkv_plan_device.default(
        *arguments,
        2,
        kinds,
        sources,
        destinations,
        misses,
        counts,
        errors,
    )

    assert torch.equal(kinds, expected[0])
    assert torch.equal(sources, expected[1])
    assert torch.equal(destinations, expected[2])
    assert torch.equal(misses, expected[3])
    assert torch.equal(counts, expected[4])
    assert not errors.any().item()
    for storage, fill in (
        (kind_storage, 99),
        (source_storage, 997),
        (destination_storage, 998),
        (miss_storage, 999),
        (count_storage, 1000),
        (error_storage, 1001),
    ):
        assert torch.equal(storage[:guard], torch.full_like(storage[:guard], fill))
        assert torch.equal(storage[-guard:], torch.full_like(storage[-guard:], fill))


def test_shadowkv_plan_device_wrapper_has_no_error_materialization():
    import inspect

    source = inspect.getsource(sgl_kernel.shadowkv_plan_device)
    assert ".cpu(" not in source
    assert ".item(" not in source
