import math

import pytest
import sgl_kernel
import sgl_kernel.shadowkv as shadowkv_module
import torch
from sglang.kernels.ops import shadowkv as shadowkv_api


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


def _value_miss_descriptor_reference(arguments, plan_reference):
    selected = arguments[0]
    kinds, _, _, miss_ordinals, counts = plan_reference
    rows, selected_capacity = selected.shape
    chunk_ids = torch.full(
        (rows, selected_capacity), -1, dtype=torch.int32, device="cuda"
    )
    for row in range(rows):
        for selected_ordinal in range(selected_capacity):
            if kinds[1, row, selected_ordinal].item() == 2:
                miss_ordinal = miss_ordinals[1, row, selected_ordinal].item()
                chunk_ids[row, miss_ordinal] = selected[row, selected_ordinal]
    return chunk_ids, counts[1, :, 1].clone()


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


def _device_placement_plan_arguments(mode):
    selected = torch.tensor(
        [[8, 5, 8, 99, 6], [4, 7, 4, 99, 9]],
        dtype=torch.int32,
        device="cuda",
    )
    selected_lengths = torch.tensor([5, 5], dtype=torch.int32, device="cuda")
    exact = torch.tensor([[99, -1], [99, -1]], dtype=torch.int32, device="cuda")
    exact_lengths = torch.tensor([1, 1], dtype=torch.int32, device="cuda")
    temporal_ids = torch.full((1, 2, 2, 3), -1, dtype=torch.int32, device="cuda")
    temporal_ids[0, 1, 0] = torch.tensor([5, 6, 8], dtype=torch.int32, device="cuda")
    temporal_ids[0, 1, 1] = torch.tensor([7, 9, 4], dtype=torch.int32, device="cuda")
    validity = torch.zeros((2, 1, 2, 2, 3), dtype=torch.uint8, device="cuda")
    publications = torch.full((2, 1, 2, 2, 3), -1, dtype=torch.int64, device="cuda")
    if mode == "all-hit":
        validity[:, 0, 1].fill_(1)
        publications[:, 0, 1].fill_(4)
    elif mode == "asymmetric":
        validity[0, 0, 1, 0] = torch.tensor([1, 0, 1], dtype=torch.uint8, device="cuda")
        validity[1, 0, 1, 0] = torch.tensor([0, 1, 1], dtype=torch.uint8, device="cuda")
        validity[0, 0, 1, 1] = torch.tensor([1, 0, 1], dtype=torch.uint8, device="cuda")
        validity[1, 0, 1, 1] = torch.tensor([0, 1, 1], dtype=torch.uint8, device="cuda")
        publications.masked_fill_(validity == 1, 4)
    elif mode != "all-miss":
        raise AssertionError(f"unknown placement mode {mode}")
    planner_arguments = (
        selected,
        selected_lengths,
        exact,
        exact_lengths,
        temporal_ids,
        validity,
        publications,
        torch.tensor([7], dtype=torch.int64, device="cuda"),
        torch.tensor([3], dtype=torch.int64, device="cuda"),
        torch.tensor([[0, 1, 0], [0, 1, 1]], dtype=torch.int32, device="cuda"),
        torch.tensor([[7, 3, 9], [7, 3, 9]], dtype=torch.int64, device="cuda"),
        torch.tensor([1, 1], dtype=torch.int32, device="cuda"),
    )
    return planner_arguments, exact, exact_lengths


def _device_placement_arguments(mode, *, parallel=False):
    planner_arguments, exact, exact_lengths = _device_placement_plan_arguments(mode)
    planner = (
        sgl_kernel.shadowkv_plan_device_v2
        if parallel
        else sgl_kernel.shadowkv_plan_device
    )
    plan = planner(*planner_arguments, plan_capacity=2)
    temporal_values = torch.empty(
        (2, 1, 2, 2, 3, 8, 128), dtype=torch.bfloat16, device="cuda"
    )
    for component in range(2):
        for layer in range(2):
            for head in range(2):
                for temporal in range(3):
                    temporal_values[component, 0, layer, head, temporal].fill_(
                        component * 1000 + layer * 100 + head * 10 + temporal + 1
                    )
    compatibility = torch.empty((2, 2, 5, 8, 128), dtype=torch.bfloat16, device="cuda")
    for component in range(2):
        for head in range(2):
            for selected_ordinal in range(5):
                compatibility[component, head, selected_ordinal].fill_(
                    -(component * 100 + head * 10 + selected_ordinal + 1)
                )
    return plan, temporal_values, compatibility, exact, exact_lengths


def _device_placement_reference(plan, temporal_values, compatibility):
    expected = torch.zeros_like(compatibility)
    temporal_chunks = temporal_values.view(-1, 8, 128)
    for component in range(plan.component_kinds.shape[0]):
        for head in range(plan.component_kinds.shape[1]):
            if plan.error_codes[head].item() != 0:
                continue
            for selected in range(plan.component_kinds.shape[2]):
                kind = plan.component_kinds[component, head, selected].item()
                if kind == 1:
                    source = plan.source_slots[component, head, selected].item()
                    expected[component, head, selected].copy_(temporal_chunks[source])
                elif kind == 2:
                    expected[component, head, selected].copy_(
                        compatibility[component, head, selected]
                    )
    return expected


def _miss_only_placement_inputs(plan, compatibility):
    reconstructed_keys = compatibility[0].clone()
    value_misses = torch.full_like(compatibility[1], 37)
    for head in range(plan.component_kinds.shape[1]):
        for selected in range(plan.component_kinds.shape[2]):
            if plan.component_kinds[1, head, selected].item() != 2:
                continue
            miss_ordinal = plan.miss_ordinals[1, head, selected].item()
            value_misses[head, miss_ordinal].copy_(compatibility[1, head, selected])
    return reconstructed_keys, value_misses


def _mapped_host_placement_inputs(plan, compatibility, prompt_chunk_capacity=128):
    heads = plan.selected_chunk_ids.shape[0]
    elements = heads * prompt_chunk_capacity * 8 * 128
    host_values = (
        torch.arange(elements, dtype=torch.int32)
        .remainder_(8192)
        .to(torch.bfloat16)
        .view(heads, prompt_chunk_capacity, 8, 128)
        .pin_memory()
    )
    matched_compatibility = compatibility.clone()
    for head in range(heads):
        for selected in range(plan.selected_chunk_ids.shape[1]):
            chunk = plan.selected_chunk_ids[head, selected].item()
            if chunk >= 0:
                matched_compatibility[1, head, selected].copy_(host_values[head, chunk])
    return host_values, matched_compatibility


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


def _a100_fused_key_case(mode, *, miss_counts=None):
    selected = torch.arange(256, dtype=torch.int32, device="cuda").repeat(8, 1)
    selected_lengths = torch.tensor(
        [256, 255, 254, 253, 252, 251, 250, 249],
        dtype=torch.int32,
        device="cuda",
    )
    exact = torch.full((8, 1), -1, dtype=torch.int32, device="cuda")
    exact_lengths = torch.zeros((8,), dtype=torch.int32, device="cuda")
    if mode == "duplicate-exact":
        selected[:, 1] = selected[:, 0]
        exact[:, 0] = selected[:, 2]
        exact_lengths.fill_(1)
    temporal_ids = (
        torch.arange(256, dtype=torch.int32, device="cuda")
        .repeat(8, 1)
        .view(1, 1, 8, 256)
    )
    validity = torch.zeros((2, 1, 1, 8, 256), dtype=torch.uint8, device="cuda")
    if miss_counts is not None:
        if mode != "boundary":
            raise AssertionError("explicit miss counts require boundary mode")
        if len(miss_counts) != 8 or any(count < 0 or count > 256 for count in miss_counts):
            raise AssertionError("A100 boundary miss counts must cover eight heads")
        validity.fill_(1)
        selected_lengths.fill_(256)
        for head, count in enumerate(miss_counts):
            validity[:, :, :, head, :count].fill_(0)
    elif mode == "all-hit":
        validity.fill_(1)
    elif mode == "mixed":
        validity[:, :, :, :, ::2].fill_(1)
    elif mode not in {"all-miss", "duplicate-exact"}:
        raise AssertionError(f"unknown fused-key mode {mode}")
    publications = torch.full((2, 1, 1, 8, 256), -1, dtype=torch.int64, device="cuda")
    publications.masked_fill_(validity == 1, 0)
    row_indices = torch.stack(
        (
            torch.zeros(8, dtype=torch.int32, device="cuda"),
            torch.zeros(8, dtype=torch.int32, device="cuda"),
            torch.arange(8, dtype=torch.int32, device="cuda"),
        ),
        dim=1,
    )
    row_generations = torch.tensor(
        [[7, 3, 1] for _ in range(8)], dtype=torch.int64, device="cuda"
    )
    plan = sgl_kernel.shadowkv_plan_device_v2(
        selected,
        selected_lengths,
        exact,
        exact_lengths,
        temporal_ids,
        validity,
        publications,
        torch.tensor([7], dtype=torch.int64, device="cuda"),
        torch.tensor([3], dtype=torch.int64, device="cuda"),
        row_indices,
        row_generations,
        torch.zeros(8, dtype=torch.int32, device="cuda"),
        plan_capacity=1,
    )
    temporal_values = (
        torch.randn((2, 1, 1, 8, 256, 8, 128), device="cuda") * 0.125
    ).to(torch.bfloat16)
    return plan, temporal_values


@pytest.mark.skipif(
    not sgl_kernel.shadowkv_a100_fused_key_kernels_available(),
    reason="the installed wheel has no SM80 fused-key child",
)
@pytest.mark.parametrize(
    "mode", ["all-hit", "mixed", "all-miss", "duplicate-exact"]
)
def test_shadowkv_a100_fused_key_matches_plan_reference(mode):
    torch.manual_seed(20260902 + len(mode))
    u = (torch.randn((2048, 160), device="cuda") * 0.0625).to(torch.bfloat16)
    sv = (torch.randn((8, 160, 128), device="cuda") * 0.0625).to(torch.bfloat16)
    inverse = 1.0 / (
        500_000.0 ** (torch.arange(0, 128, 2, device="cuda", dtype=torch.float32) / 128)
    )
    positions = torch.arange(2048, dtype=torch.float32, device="cuda")
    angles = positions[:, None] * inverse
    cosine = angles.cos().contiguous()
    sine = angles.sin().contiguous()
    plan, temporal_values = _a100_fused_key_case(mode)
    output_shape = (2, 8, 256, 8, 128)
    output_elements = math.prod(output_shape)
    guard_elements = 256
    output_storage = torch.full(
        (guard_elements + output_elements + guard_elements,),
        23,
        dtype=torch.bfloat16,
        device="cuda",
    )
    output = output_storage[
        guard_elements : guard_elements + output_elements
    ].view(output_shape)
    gathered_shape = (8, 2048, 160)
    gathered_elements = math.prod(gathered_shape)
    gathered_storage = torch.full(
        (guard_elements + gathered_elements + guard_elements,),
        37,
        dtype=torch.bfloat16,
        device="cuda",
    )
    gathered_u = gathered_storage[
        guard_elements : guard_elements + gathered_elements
    ].view(gathered_shape)

    selected_positions = (
        plan.selected_chunk_ids.to(torch.int64)[..., None] * 8
        + torch.arange(8, dtype=torch.int64, device="cuda")
    ).reshape(8, 2048)
    pre_rope = sgl_kernel.shadowkv_reconstruct(u, sv, selected_positions).float()
    selected_cosine = cosine[selected_positions]
    selected_sine = sine[selected_positions]
    expected_misses = (
        pre_rope * torch.cat((selected_cosine, selected_cosine), dim=-1)
        + torch.cat((-pre_rope[..., 64:], pre_rope[..., :64]), dim=-1)
        * torch.cat((selected_sine, selected_sine), dim=-1)
    ).to(torch.bfloat16)
    expected = torch.zeros((8, 256, 8, 128), dtype=torch.bfloat16, device="cuda")
    temporal_chunks = temporal_values.view(2, -1, 8, 128)
    for head in range(8):
        for selected in range(256):
            kind = plan.component_kinds[0, head, selected].item()
            if kind == 1:
                source = plan.source_slots[0, head, selected].item()
                expected[head, selected].copy_(temporal_chunks[0, source])
            elif kind == 2:
                start = selected * 8
                expected[head, selected].copy_(expected_misses[head, start : start + 8])

    actual = sgl_kernel.shadowkv_fused_key_a100(
        u,
        sv,
        gathered_u,
        cosine,
        sine,
        plan.component_kinds,
        plan.source_slots,
        plan.destination_slots,
        plan.miss_ordinals,
        plan.selected_chunk_ids,
        plan.selected_lengths,
        plan.plan_slots,
        plan.error_codes,
        temporal_values,
        plan_capacity=1,
        out=output,
    )
    repeated = output.clone()
    sgl_kernel.shadowkv_fused_key_a100(
        u,
        sv,
        gathered_u,
        cosine,
        sine,
        plan.component_kinds,
        plan.source_slots,
        plan.destination_slots,
        plan.miss_ordinals,
        plan.selected_chunk_ids,
        plan.selected_lengths,
        plan.plan_slots,
        plan.error_codes,
        temporal_values,
        plan_capacity=1,
        out=repeated,
    )

    assert actual.data_ptr() == output[0].data_ptr()
    assert torch.count_nonzero(plan.error_codes).item() == 0
    assert torch.equal(output[1], torch.full_like(output[1], 23))
    assert torch.equal(repeated[0], output[0])
    torch.testing.assert_close(output[0], expected, rtol=2e-3, atol=2e-3)
    assert torch.equal(
        output_storage[:guard_elements],
        torch.full_like(output_storage[:guard_elements], 23),
    )
    assert torch.equal(
        output_storage[-guard_elements:],
        torch.full_like(output_storage[-guard_elements:], 23),
    )
    assert torch.equal(
        gathered_storage[:guard_elements],
        torch.full_like(gathered_storage[:guard_elements], 37),
    )
    assert torch.equal(
        gathered_storage[-guard_elements:],
        torch.full_like(gathered_storage[-guard_elements:], 37),
    )


@pytest.mark.skipif(
    not sgl_kernel.shadowkv_a100_fused_key_kernels_available(),
    reason="the installed wheel has no SM80 fused-key child",
)
def test_shadowkv_a100_miss_only_boundaries_match_retained_full_bmm_control():
    torch.manual_seed(20261004)
    miss_counts = (0, 1, 15, 16, 17, 255, 256, 127)
    plan, temporal_values = _a100_fused_key_case(
        "boundary", miss_counts=miss_counts
    )
    u = (torch.randn((2048, 160), device="cuda") * 0.0625).to(torch.bfloat16)
    sv = (torch.randn((8, 160, 128), device="cuda") * 0.0625).to(torch.bfloat16)
    inverse = 1.0 / (
        500_000.0
        ** (torch.arange(0, 128, 2, device="cuda", dtype=torch.float32) / 128)
    )
    angles = torch.arange(2048, dtype=torch.float32, device="cuda")[:, None] * inverse
    cosine = angles.cos().contiguous()
    sine = angles.sin().contiguous()
    full_workspace = torch.empty(
        (8, 2048, 160), dtype=torch.bfloat16, device="cuda"
    )
    full_output = torch.full(
        (2, 8, 256, 8, 128), 41, dtype=torch.bfloat16, device="cuda"
    )
    miss_workspace = torch.full_like(full_workspace, 43)
    miss_output = torch.full_like(full_output, 47)
    exact_gathered = torch.full_like(full_workspace, 61)
    exact_reconstructed = torch.full(
        (8, 2048, 128), 67, dtype=torch.bfloat16, device="cuda"
    )
    exact_output = torch.full_like(full_output, 71)
    host_miss_counts = torch.empty((8,), dtype=torch.int32, pin_memory=True)
    exact_ready = torch.cuda.Event()
    exact_ready.record()
    exact_ready.synchronize()
    torch.ops.sgl_kernel.shadowkv_prepare_exact_miss_gemm_sm80_a100_v1.default(u)
    mapped_miss_counts = int(
        torch.ops.sgl_kernel.shadowkv_resolve_miss_count_pointer_sm80_a100_v1.default(
            host_miss_counts,
            torch.cuda.current_device(),
        )
    )

    torch.ops.sgl_kernel.shadowkv_fused_key_sm80_a100_v2.default(
        u,
        sv,
        full_workspace,
        cosine,
        sine,
        plan.component_kinds,
        plan.source_slots,
        plan.destination_slots,
        plan.miss_ordinals,
        plan.selected_chunk_ids,
        plan.selected_lengths,
        plan.plan_slots,
        plan.error_codes,
        temporal_values,
        1,
        full_output,
    )
    torch.ops.sgl_kernel.shadowkv_fused_key_sm80_a100_v3.default(
        u,
        sv,
        miss_workspace,
        cosine,
        sine,
        plan.component_kinds,
        plan.source_slots,
        plan.destination_slots,
        plan.miss_ordinals,
        plan.selected_chunk_ids,
        plan.selected_lengths,
        plan.plan_slots,
        plan.error_codes,
        temporal_values,
        1,
        miss_output,
    )
    torch.ops.sgl_kernel.shadowkv_fused_key_sm80_a100_v4.default(
        u,
        sv,
        exact_gathered,
        exact_reconstructed,
        cosine,
        sine,
        plan.component_kinds,
        plan.source_slots,
        plan.destination_slots,
        plan.miss_ordinals,
        plan.selected_chunk_ids,
        plan.selected_lengths,
        plan.plan_slots,
        plan.error_codes,
        temporal_values,
        host_miss_counts,
        mapped_miss_counts,
        exact_ready.cuda_event,
        1,
        exact_output,
    )

    assert not plan.error_codes.any().item()
    torch.testing.assert_close(miss_output[0], full_output[0], rtol=2e-3, atol=2e-3)
    assert torch.equal(exact_output[0], full_output[0])
    assert torch.equal(miss_output[1], torch.full_like(miss_output[1], 47))
    assert torch.equal(exact_output[1], torch.full_like(exact_output[1], 71))
    assert tuple(host_miss_counts.tolist()) == miss_counts
    assert torch.equal(miss_workspace[0], torch.full_like(miss_workspace[0], 43))
    for head, expected_misses in enumerate(miss_counts):
        assert (
            torch.count_nonzero(plan.component_kinds[0, head] == 2).item()
            == expected_misses
        )
        hit_rows = plan.component_kinds[0, head] == 1
        assert torch.equal(miss_output[0, head, hit_rows], full_output[0, head, hit_rows])


@pytest.mark.skipif(
    not sgl_kernel.shadowkv_a100_fused_key_kernels_available(),
    reason="the installed wheel has no SM80 fused-key child",
)
def test_shadowkv_a100_exact_all_hit_skips_compaction_and_gemm():
    torch.manual_seed(20261007)
    plan, temporal_values = _a100_fused_key_case("all-hit")
    u = torch.zeros((2048, 160), dtype=torch.bfloat16, device="cuda")
    sv = torch.zeros((8, 160, 128), dtype=torch.bfloat16, device="cuda")
    cosine = torch.ones((2048, 64), dtype=torch.float32, device="cuda")
    sine = torch.zeros_like(cosine)
    full_gathered = torch.empty(
        (8, 2048, 160), dtype=torch.bfloat16, device="cuda"
    )
    full_output = torch.full(
        (2, 8, 256, 8, 128), 73, dtype=torch.bfloat16, device="cuda"
    )
    exact_gathered = torch.full_like(full_gathered, 79)
    exact_reconstructed = torch.full(
        (8, 2048, 128), 83, dtype=torch.bfloat16, device="cuda"
    )
    exact_output = torch.full_like(full_output, 89)
    host_miss_counts = torch.empty((8,), dtype=torch.int32, pin_memory=True)
    exact_ready = torch.cuda.Event()
    exact_ready.record()
    exact_ready.synchronize()
    torch.ops.sgl_kernel.shadowkv_prepare_exact_miss_gemm_sm80_a100_v1.default(u)
    mapped_miss_counts = int(
        torch.ops.sgl_kernel.shadowkv_resolve_miss_count_pointer_sm80_a100_v1.default(
            host_miss_counts,
            torch.cuda.current_device(),
        )
    )

    torch.ops.sgl_kernel.shadowkv_fused_key_sm80_a100_v2.default(
        u,
        sv,
        full_gathered,
        cosine,
        sine,
        plan.component_kinds,
        plan.source_slots,
        plan.destination_slots,
        plan.miss_ordinals,
        plan.selected_chunk_ids,
        plan.selected_lengths,
        plan.plan_slots,
        plan.error_codes,
        temporal_values,
        1,
        full_output,
    )
    torch.ops.sgl_kernel.shadowkv_fused_key_sm80_a100_v4.default(
        u,
        sv,
        exact_gathered,
        exact_reconstructed,
        cosine,
        sine,
        plan.component_kinds,
        plan.source_slots,
        plan.destination_slots,
        plan.miss_ordinals,
        plan.selected_chunk_ids,
        plan.selected_lengths,
        plan.plan_slots,
        plan.error_codes,
        temporal_values,
        host_miss_counts,
        mapped_miss_counts,
        exact_ready.cuda_event,
        1,
        exact_output,
    )

    assert not plan.error_codes.any().item()
    assert torch.equal(exact_output[0], full_output[0])
    assert tuple(host_miss_counts.tolist()) == (0,) * 8
    assert torch.equal(exact_gathered, torch.full_like(exact_gathered, 79))
    assert torch.equal(
        exact_reconstructed, torch.full_like(exact_reconstructed, 83)
    )


@pytest.mark.skipif(
    not sgl_kernel.shadowkv_a100_fused_key_kernels_available(),
    reason="the installed wheel has no SM80 fused-key child",
)
@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate-miss-ordinal",
        "missing-miss-ordinal",
        "invalid-plan-slot",
        "oversized-selected-length",
        "invalid-hit-source",
        "invalid-selected-chunk",
    ],
)
def test_shadowkv_a100_miss_only_rejects_invalid_plan_before_write(mutation):
    torch.manual_seed(20261005)
    mode = "all-hit" if mutation == "invalid-hit-source" else "all-miss"
    plan, temporal_values = _a100_fused_key_case(mode)
    if mutation == "duplicate-miss-ordinal":
        plan.miss_ordinals[0, 0, 1] = 0
    elif mutation == "missing-miss-ordinal":
        plan.miss_ordinals[0, 0, 1] = 2
    elif mutation == "invalid-plan-slot":
        plan.plan_slots[0] = -1
    elif mutation == "oversized-selected-length":
        plan.selected_lengths[0] = 257
    elif mutation == "invalid-hit-source":
        plan.source_slots[0, 0, 0] = 1 << 30
    elif mutation == "invalid-selected-chunk":
        plan.selected_chunk_ids[0, 0] = 256
    else:
        raise AssertionError(f"unknown invalid-plan mutation {mutation}")
    u = torch.zeros((2048, 160), dtype=torch.bfloat16, device="cuda")
    sv = torch.zeros((8, 160, 128), dtype=torch.bfloat16, device="cuda")
    cosine = torch.ones((2048, 64), dtype=torch.float32, device="cuda")
    sine = torch.zeros_like(cosine)
    gathered_u = torch.full(
        (8, 2048, 160), 53, dtype=torch.bfloat16, device="cuda"
    )
    output = torch.full(
        (2, 8, 256, 8, 128), 59, dtype=torch.bfloat16, device="cuda"
    )
    exact_gathered = torch.full_like(gathered_u, 61)
    exact_reconstructed = torch.full(
        (8, 2048, 128), 67, dtype=torch.bfloat16, device="cuda"
    )
    exact_output = torch.full_like(output, 71)
    host_miss_counts = torch.empty((8,), dtype=torch.int32, pin_memory=True)
    exact_ready = torch.cuda.Event()
    exact_ready.record()
    exact_ready.synchronize()
    torch.ops.sgl_kernel.shadowkv_prepare_exact_miss_gemm_sm80_a100_v1.default(u)
    mapped_miss_counts = int(
        torch.ops.sgl_kernel.shadowkv_resolve_miss_count_pointer_sm80_a100_v1.default(
            host_miss_counts,
            torch.cuda.current_device(),
        )
    )

    with pytest.raises(
        RuntimeError, match="exact A100 miss count was not published safely"
    ):
        torch.ops.sgl_kernel.shadowkv_fused_key_sm80_a100_v4.default(
            u,
            sv,
            exact_gathered,
            exact_reconstructed,
            cosine,
            sine,
            plan.component_kinds,
            plan.source_slots,
            plan.destination_slots,
            plan.miss_ordinals,
            plan.selected_chunk_ids,
            plan.selected_lengths,
            plan.plan_slots,
            plan.error_codes,
            temporal_values,
            host_miss_counts,
            mapped_miss_counts,
            exact_ready.cuda_event,
            1,
            exact_output,
        )
    assert torch.equal(exact_output[0, 0], torch.full_like(exact_output[0, 0], 71))
    plan.error_codes.zero_()

    torch.ops.sgl_kernel.shadowkv_fused_key_sm80_a100_v3.default(
        u,
        sv,
        gathered_u,
        cosine,
        sine,
        plan.component_kinds,
        plan.source_slots,
        plan.destination_slots,
        plan.miss_ordinals,
        plan.selected_chunk_ids,
        plan.selected_lengths,
        plan.plan_slots,
        plan.error_codes,
        temporal_values,
        1,
        output,
    )

    assert plan.error_codes[0].item() != 0
    assert torch.equal(output[0, 0], torch.full_like(output[0, 0], 59))
    assert torch.equal(gathered_u[0], torch.full_like(gathered_u[0], 53))


@pytest.mark.skipif(
    not sgl_kernel.shadowkv_a100_fused_key_kernels_available(),
    reason="the installed wheel has no SM80 fused-key child",
)
def test_shadowkv_a100_miss_only_rejects_misaligned_destination_before_launch():
    plan, temporal_values = _a100_fused_key_case("all-hit")
    u = torch.zeros((2048, 160), dtype=torch.bfloat16, device="cuda")
    sv = torch.zeros((8, 160, 128), dtype=torch.bfloat16, device="cuda")
    cosine = torch.ones((2048, 64), dtype=torch.float32, device="cuda")
    sine = torch.zeros_like(cosine)
    gathered_u = torch.empty(
        (8, 2048, 160), dtype=torch.bfloat16, device="cuda"
    )
    output_elements = 2 * 8 * 256 * 8 * 128
    output_storage = torch.empty(
        (output_elements + 1,), dtype=torch.bfloat16, device="cuda"
    )
    output = output_storage[1:].view(2, 8, 256, 8, 128)

    with pytest.raises(RuntimeError, match="destination_key_values must be 16-byte aligned"):
        torch.ops.sgl_kernel.shadowkv_fused_key_sm80_a100_v3.default(
            u,
            sv,
            gathered_u,
            cosine,
            sine,
            plan.component_kinds,
            plan.source_slots,
            plan.destination_slots,
            plan.miss_ordinals,
            plan.selected_chunk_ids,
            plan.selected_lengths,
            plan.plan_slots,
            plan.error_codes,
            temporal_values,
            1,
            output,
        )


@pytest.mark.skipif(
    not sgl_kernel.shadowkv_a100_fused_key_kernels_available(),
    reason="the installed wheel has no SM80 fused-key child",
)
@pytest.mark.parametrize("mode", ["all-hit", "mixed", "all-miss"])
def test_shadowkv_a100_combined_mapped_value_and_fused_key_matches_split(mode):
    torch.manual_seed(20260930 + len(mode))
    u = (torch.randn((2048, 160), device="cuda") * 0.0625).to(torch.bfloat16)
    sv = (torch.randn((8, 160, 128), device="cuda") * 0.0625).to(torch.bfloat16)
    inverse = 1.0 / (
        500_000.0 ** (torch.arange(0, 128, 2, device="cuda", dtype=torch.float32) / 128)
    )
    positions = torch.arange(2048, dtype=torch.float32, device="cuda")
    angles = positions[:, None] * inverse
    cosine = angles.cos().contiguous()
    sine = angles.sin().contiguous()
    plan, temporal_values = _a100_fused_key_case(mode)
    descriptor_generation = torch.tensor([7], dtype=torch.int64, device="cuda")
    descriptor_validity = torch.ones((1,), dtype=torch.uint8, device="cuda")
    host_values = (
        torch.arange(8 * 256 * 8 * 128, dtype=torch.int32)
        .remainder_(8192)
        .to(torch.bfloat16)
        .view(8, 256, 8, 128)
        .pin_memory()
    )
    mapped_region = sgl_kernel.shadowkv_resolve_mapped_host_region(
        host_values,
        device="cuda",
    )
    gathered_u = torch.empty((8, 2048, 160), dtype=torch.bfloat16, device="cuda")
    expected = torch.full(
        (2, 8, 256, 8, 128),
        29,
        dtype=torch.bfloat16,
        device="cuda",
    )
    actual = torch.full_like(expected, 31)

    sgl_kernel.shadowkv_fused_key_a100(
        u,
        sv,
        gathered_u,
        cosine,
        sine,
        plan.component_kinds,
        plan.source_slots,
        plan.destination_slots,
        plan.miss_ordinals,
        plan.selected_chunk_ids,
        plan.selected_lengths,
        plan.plan_slots,
        plan.error_codes,
        temporal_values,
        plan_capacity=1,
        out=expected,
    )
    sgl_kernel.shadowkv_place_value_mapped_host_a100(
        plan.component_kinds,
        plan.source_slots,
        plan.destination_slots,
        plan.miss_ordinals,
        plan.selected_chunk_ids,
        plan.selected_lengths,
        plan.plan_slots,
        plan.error_codes,
        temporal_values,
        plan.value_miss_chunk_ids,
        plan.value_miss_lengths,
        descriptor_generation,
        descriptor_validity,
        mapped_region,
        prompt_tokens=2048,
        expected_generation=7,
        plan_capacity=1,
        out=expected,
    )

    caller_stream = torch.cuda.current_stream()
    mapped_stream = torch.cuda.Stream()
    reconstruction_stream = torch.cuda.Stream()
    mapped_stream.wait_stream(caller_stream)
    reconstruction_stream.wait_stream(caller_stream)
    with torch.cuda.stream(mapped_stream):
        actual_keys, actual_values = sgl_kernel.shadowkv_fused_key_mapped_value_a100(
            u,
            sv,
            gathered_u,
            cosine,
            sine,
            plan.component_kinds,
            plan.source_slots,
            plan.destination_slots,
            plan.miss_ordinals,
            plan.selected_chunk_ids,
            plan.selected_lengths,
            plan.plan_slots,
            plan.error_codes,
            temporal_values,
            plan.value_miss_chunk_ids,
            plan.value_miss_lengths,
            descriptor_generation,
            descriptor_validity,
            mapped_region,
            reconstruction_stream,
            prompt_tokens=2048,
            expected_generation=7,
            plan_capacity=1,
            out=actual,
        )
    caller_stream.wait_stream(mapped_stream)
    caller_stream.wait_stream(reconstruction_stream)

    assert actual_keys.data_ptr() == actual[0].data_ptr()
    assert actual_values.data_ptr() == actual[1].data_ptr()
    assert torch.equal(actual, expected)

    exact_expected = torch.full_like(expected, 37)
    exact_actual = torch.full_like(expected, 41)
    exact_gathered = torch.empty_like(gathered_u)
    exact_reconstructed = torch.empty(
        (8, 2048, 128), dtype=torch.bfloat16, device="cuda"
    )
    host_miss_counts = torch.empty((8,), dtype=torch.int32, pin_memory=True)
    exact_ready = torch.cuda.Event()
    exact_ready.record()
    exact_ready.synchronize()
    torch.ops.sgl_kernel.shadowkv_prepare_exact_miss_gemm_sm80_a100_v1.default(u)
    mapped_miss_counts = int(
        torch.ops.sgl_kernel.shadowkv_resolve_miss_count_pointer_sm80_a100_v1.default(
            host_miss_counts,
            torch.cuda.current_device(),
        )
    )
    torch.ops.sgl_kernel.shadowkv_fused_key_sm80_a100_v2.default(
        u,
        sv,
        exact_gathered,
        cosine,
        sine,
        plan.component_kinds,
        plan.source_slots,
        plan.destination_slots,
        plan.miss_ordinals,
        plan.selected_chunk_ids,
        plan.selected_lengths,
        plan.plan_slots,
        plan.error_codes,
        temporal_values,
        1,
        exact_expected,
    )
    sgl_kernel.shadowkv_place_value_mapped_host_a100(
        plan.component_kinds,
        plan.source_slots,
        plan.destination_slots,
        plan.miss_ordinals,
        plan.selected_chunk_ids,
        plan.selected_lengths,
        plan.plan_slots,
        plan.error_codes,
        temporal_values,
        plan.value_miss_chunk_ids,
        plan.value_miss_lengths,
        descriptor_generation,
        descriptor_validity,
        mapped_region,
        prompt_tokens=2048,
        expected_generation=7,
        plan_capacity=1,
        out=exact_expected,
    )
    exact_mapped_stream = torch.cuda.Stream()
    exact_reconstruction_stream = torch.cuda.Stream()
    exact_mapped_stream.wait_stream(caller_stream)
    exact_reconstruction_stream.wait_stream(caller_stream)
    with torch.cuda.stream(exact_mapped_stream):
        torch.ops.sgl_kernel.shadowkv_fused_key_mapped_value_sm80_a100_v7.default(
            u,
            sv,
            exact_gathered,
            exact_reconstructed,
            cosine,
            sine,
            plan.component_kinds,
            plan.source_slots,
            plan.destination_slots,
            plan.miss_ordinals,
            plan.selected_chunk_ids,
            plan.selected_lengths,
            plan.plan_slots,
            plan.error_codes,
            temporal_values,
            plan.value_miss_chunk_ids,
            plan.value_miss_lengths,
            descriptor_generation,
            descriptor_validity,
            mapped_region.device_pointer,
            mapped_region.byte_length,
            mapped_region.prompt_chunk_capacity,
            2048,
            7,
            host_miss_counts,
            mapped_miss_counts,
            exact_ready.cuda_event,
            1,
            exact_reconstruction_stream.cuda_stream,
            exact_actual,
        )
    caller_stream.wait_stream(exact_mapped_stream)
    caller_stream.wait_stream(exact_reconstruction_stream)
    assert torch.equal(exact_actual, exact_expected)
    assert tuple(host_miss_counts.tolist()) == tuple(
        int(torch.count_nonzero(plan.component_kinds[0, head] == 2).item())
        for head in range(8)
    )


@pytest.mark.skipif(
    not sgl_kernel.shadowkv_a100_fused_key_kernels_available(),
    reason="the installed wheel has no SM80 fused-key child",
)
@pytest.mark.parametrize("mode", ["all-hit", "mixed", "all-miss"])
def test_shadowkv_a100_value_only_placement_matches_plan_reference(mode):
    torch.manual_seed(20260920 + len(mode))
    plan, temporal_values = _a100_fused_key_case(mode)
    compatibility = torch.randn(
        (2, 8, 256, 8, 128),
        dtype=torch.bfloat16,
        device="cuda",
    )
    expected = torch.zeros_like(compatibility[1])
    temporal_chunks = temporal_values.view(2, -1, 8, 128)
    for head in range(8):
        for selected in range(256):
            kind = plan.component_kinds[1, head, selected].item()
            if kind == 1:
                source = plan.source_slots[1, head, selected].item()
                expected[head, selected].copy_(temporal_chunks[1, source - 2048])
            elif kind == 2:
                expected[head, selected].copy_(compatibility[1, head, selected])

    guard = 256
    elements = compatibility.numel()
    guarded = torch.full(
        (guard + elements + guard,),
        19,
        dtype=torch.bfloat16,
        device="cuda",
    )
    output = guarded[guard : guard + elements].view_as(compatibility)
    sgl_kernel.shadowkv_place_value_a100(
        plan.component_kinds,
        plan.source_slots,
        plan.destination_slots,
        plan.selected_lengths,
        plan.plan_slots,
        plan.error_codes,
        temporal_values,
        compatibility,
        plan_capacity=1,
        out=output,
    )

    assert not plan.error_codes.any().item()
    assert torch.equal(output[0], torch.full_like(output[0], 19))
    assert torch.equal(output[1], expected)
    assert torch.equal(guarded[:guard], torch.full_like(guarded[:guard], 19))
    assert torch.equal(guarded[-guard:], torch.full_like(guarded[-guard:], 19))


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
    parallel = sgl_kernel.shadowkv_plan_device_v2(*arguments, plan_capacity=2)
    parallel_repeated = sgl_kernel.shadowkv_plan_device_v2(*arguments, plan_capacity=2)
    expected_value_misses = _value_miss_descriptor_reference(arguments, expected)

    assert not actual.error_codes.any().item()
    assert torch.equal(actual.component_kinds, expected[0])
    assert torch.equal(actual.source_slots, expected[1])
    assert torch.equal(actual.destination_slots, expected[2])
    assert torch.equal(actual.miss_ordinals, expected[3])
    assert torch.equal(actual.counts, expected[4])
    assert torch.equal(actual.component_kinds, repeated.component_kinds)
    assert actual.selected_chunk_ids.data_ptr() == arguments[0].data_ptr()
    assert actual.row_indices.data_ptr() == arguments[9].data_ptr()
    assert actual.plan_slots.data_ptr() == arguments[11].data_ptr()
    assert torch.equal(parallel.component_kinds, expected[0])
    assert torch.equal(parallel.source_slots, expected[1])
    assert torch.equal(parallel.destination_slots, expected[2])
    assert torch.equal(parallel.miss_ordinals, expected[3])
    assert torch.equal(parallel.counts, expected[4])
    assert torch.equal(parallel.value_miss_chunk_ids, expected_value_misses[0])
    assert torch.equal(parallel.value_miss_lengths, expected_value_misses[1])
    assert torch.equal(
        parallel.value_miss_chunk_ids, parallel_repeated.value_miss_chunk_ids
    )
    assert torch.equal(parallel.component_kinds, parallel_repeated.component_kinds)


def test_shadowkv_publish_value_descriptor_sets_generation_before_validity():
    generation = torch.full((1,), -1, dtype=torch.int64, device="cuda")
    validity = torch.zeros((1,), dtype=torch.uint8, device="cuda")

    sgl_kernel.shadowkv_publish_value_descriptor(
        generation,
        validity,
        generation=17,
    )

    assert generation.item() == 17
    assert validity.item() == 1


@pytest.mark.parametrize("mode", ["all-hit", "all-miss", "asymmetric"])
def test_shadowkv_place_device_populates_each_stable_destination_once(mode):
    plan, temporal_values, compatibility, exact, exact_lengths = (
        _device_placement_arguments(mode)
    )
    expected = _device_placement_reference(plan, temporal_values, compatibility)
    guard_elements = 256
    output_elements = compatibility.numel()
    guarded = torch.full(
        (output_elements + 2 * guard_elements,),
        13,
        dtype=torch.bfloat16,
        device="cuda",
    )
    output = guarded.narrow(0, guard_elements, output_elements).view_as(compatibility)
    compatibility_before = compatibility.clone()

    actual = sgl_kernel.shadowkv_place_device(
        plan.component_kinds,
        plan.source_slots,
        plan.destination_slots,
        plan.plan_slots,
        plan.error_codes,
        temporal_values,
        compatibility,
        output,
        plan_capacity=2,
    )

    assert actual.data_ptr() == output.data_ptr()
    assert torch.equal(actual, expected)
    assert torch.equal(compatibility, compatibility_before)
    assert torch.equal(
        guarded[:guard_elements],
        torch.full_like(guarded[:guard_elements], 13),
    )
    assert torch.equal(
        guarded[-guard_elements:],
        torch.full_like(guarded[-guard_elements:], 13),
    )
    active = plan.component_kinds >= 1
    assert active.sum().item() == 12
    assert torch.count_nonzero(actual[active]).item() == active.sum().item() * 8 * 128
    assert not torch.count_nonzero(actual[~active]).item()

    sgl_kernel.shadowkv_publish_device(
        plan.selected_chunk_ids,
        plan.selected_lengths,
        exact,
        exact_lengths,
        plan.row_indices,
        plan.row_generations,
        plan.error_codes,
        actual,
        torch.tensor([7], dtype=torch.int64, device="cuda"),
        torch.tensor([3], dtype=torch.int64, device="cuda"),
        temporal_ids := torch.full((1, 2, 2, 3), -1, dtype=torch.int32, device="cuda"),
        published_values := torch.zeros_like(temporal_values),
        publication_generations := torch.full(
            (2, 1, 2, 2, 3), -1, dtype=torch.int64, device="cuda"
        ),
        published_validity := torch.zeros(
            (2, 1, 2, 2, 3), dtype=torch.uint8, device="cuda"
        ),
    )
    assert torch.equal(
        temporal_ids[0, 1],
        torch.tensor([[8, 5, 6], [4, 7, 9]], dtype=torch.int32, device="cuda"),
    )
    assert published_validity[:, 0, 1].all().item()
    assert torch.equal(
        publication_generations[:, 0, 1],
        torch.full((2, 2, 3), 9, dtype=torch.int64, device="cuda"),
    )
    assert torch.equal(published_values[0, 0, 1, 0, 0], actual[0, 0, 0])
    assert torch.equal(published_values[0, 0, 1, 0, 1], actual[0, 0, 1])
    assert torch.equal(published_values[0, 0, 1, 0, 2], actual[0, 0, 4])
    assert torch.equal(published_values[1, 0, 1, 1, 0], actual[1, 1, 0])
    assert torch.equal(published_values[1, 0, 1, 1, 1], actual[1, 1, 1])
    assert torch.equal(published_values[1, 0, 1, 1, 2], actual[1, 1, 4])


@pytest.mark.parametrize("mode", ["all-hit", "all-miss", "asymmetric"])
def test_shadowkv_place_device_miss_only_matches_full_selected_control(mode):
    plan, temporal_values, compatibility, _, _ = _device_placement_arguments(
        mode, parallel=True
    )
    reconstructed_keys, value_misses = _miss_only_placement_inputs(plan, compatibility)
    expected = _device_placement_reference(plan, temporal_values, compatibility)
    guard_elements = 256
    output_elements = compatibility.numel()
    guarded = torch.full(
        (output_elements + 2 * guard_elements,),
        41,
        dtype=torch.bfloat16,
        device="cuda",
    )
    output = guarded.narrow(0, guard_elements, output_elements).view_as(compatibility)
    generation = torch.tensor([11], dtype=torch.int64, device="cuda")
    validity_flag = torch.ones((1,), dtype=torch.uint8, device="cuda")
    value_misses_before = value_misses.clone()

    actual = sgl_kernel.shadowkv_place_device_miss_only(
        plan.component_kinds,
        plan.source_slots,
        plan.destination_slots,
        plan.miss_ordinals,
        plan.selected_chunk_ids,
        plan.plan_slots,
        plan.error_codes,
        temporal_values,
        reconstructed_keys,
        plan.value_miss_chunk_ids,
        plan.value_miss_lengths,
        generation,
        validity_flag,
        value_misses,
        output,
        expected_generation=11,
        plan_capacity=2,
    )

    assert actual.data_ptr() == output.data_ptr()
    assert torch.equal(actual, expected)
    assert torch.equal(value_misses, value_misses_before)
    assert torch.equal(
        guarded[:guard_elements], torch.full_like(guarded[:guard_elements], 41)
    )
    assert torch.equal(
        guarded[-guard_elements:], torch.full_like(guarded[-guard_elements:], 41)
    )
    expected_misses = plan.counts[1, :, 1].sum().item()
    assert plan.value_miss_lengths.sum().item() == expected_misses
    if mode == "all-hit":
        assert expected_misses == 0


def test_shadowkv_place_device_miss_only_rejects_stale_descriptor():
    plan, temporal_values, compatibility, _, _ = _device_placement_arguments("all-miss")
    reconstructed_keys, value_misses = _miss_only_placement_inputs(plan, compatibility)
    output = torch.full_like(compatibility, 43)
    descriptor_ids = torch.full(
        plan.selected_chunk_ids.shape,
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    descriptor_lengths = torch.zeros(
        plan.selected_chunk_ids.shape[0], dtype=torch.int32, device="cuda"
    )

    sgl_kernel.shadowkv_place_device_miss_only(
        plan.component_kinds,
        plan.source_slots,
        plan.destination_slots,
        plan.miss_ordinals,
        plan.selected_chunk_ids,
        plan.plan_slots,
        plan.error_codes,
        temporal_values,
        reconstructed_keys,
        descriptor_ids,
        descriptor_lengths,
        torch.tensor([10], dtype=torch.int64, device="cuda"),
        torch.ones((1,), dtype=torch.uint8, device="cuda"),
        value_misses,
        output,
        expected_generation=11,
        plan_capacity=2,
    )

    assert not torch.count_nonzero(output).item()
    assert plan.error_codes.cpu().tolist() == [8, 8]


@pytest.mark.parametrize("mode", ["all-hit", "all-miss", "asymmetric"])
def test_shadowkv_place_device_mapped_host_matches_full_selected_control(mode):
    plan, temporal_values, compatibility, _, _ = _device_placement_arguments(
        mode, parallel=True
    )
    host_values, compatibility = _mapped_host_placement_inputs(plan, compatibility)
    mapped = sgl_kernel.shadowkv_resolve_mapped_host_region(
        host_values,
        device=torch.device("cuda"),
    )
    reconstructed_keys = compatibility[0].clone()
    expected = _device_placement_reference(plan, temporal_values, compatibility)
    guard_elements = 256
    output_elements = compatibility.numel()
    guarded = torch.full(
        (output_elements + 2 * guard_elements,),
        53,
        dtype=torch.bfloat16,
        device="cuda",
    )
    output = guarded.narrow(0, guard_elements, output_elements).view_as(compatibility)

    actual = sgl_kernel.shadowkv_place_device_mapped_host(
        plan.component_kinds,
        plan.source_slots,
        plan.destination_slots,
        plan.miss_ordinals,
        plan.selected_chunk_ids,
        plan.plan_slots,
        plan.error_codes,
        temporal_values,
        reconstructed_keys,
        plan.value_miss_chunk_ids,
        plan.value_miss_lengths,
        torch.tensor([13], dtype=torch.int64, device="cuda"),
        torch.ones((1,), dtype=torch.uint8, device="cuda"),
        mapped,
        output,
        prompt_tokens=128 * 8,
        expected_generation=13,
        plan_capacity=2,
    )
    repeated = torch.empty_like(output)
    sgl_kernel.shadowkv_place_device_mapped_host(
        plan.component_kinds,
        plan.source_slots,
        plan.destination_slots,
        plan.miss_ordinals,
        plan.selected_chunk_ids,
        plan.plan_slots,
        plan.error_codes,
        temporal_values,
        reconstructed_keys,
        plan.value_miss_chunk_ids,
        plan.value_miss_lengths,
        torch.tensor([13], dtype=torch.int64, device="cuda"),
        torch.ones((1,), dtype=torch.uint8, device="cuda"),
        mapped,
        repeated,
        prompt_tokens=128 * 8,
        expected_generation=13,
        plan_capacity=2,
    )

    assert mapped.values is host_values
    assert mapped.device_pointer > 0
    assert mapped.device_pointer % 16 == 0
    assert actual.data_ptr() == output.data_ptr()
    assert torch.equal(actual, expected)
    assert torch.equal(repeated, expected)
    assert torch.equal(actual[..., -1], expected[..., -1])
    assert torch.equal(
        guarded[:guard_elements], torch.full_like(guarded[:guard_elements], 53)
    )
    assert torch.equal(
        guarded[-guard_elements:], torch.full_like(guarded[-guard_elements:], 53)
    )


def test_shadowkv_mapped_host_resolver_rejects_pageable_and_misaligned_storage():
    pageable = torch.empty((2, 8, 8, 128), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="page-locked"):
        sgl_kernel.shadowkv_resolve_mapped_host_region(pageable, device="cuda")
    with pytest.raises(RuntimeError, match="page-locked"):
        torch.ops.sgl_kernel.shadowkv_resolve_mapped_host_pointer_generic_aot_v1.default(
            pageable,
            torch.cuda.current_device(),
        )

    elements = 2 * 8 * 8 * 128
    allocation = torch.empty((elements + 1,), dtype=torch.bfloat16, pin_memory=True)
    misaligned = allocation[1:].view(2, 8, 8, 128)
    assert misaligned.is_contiguous() and misaligned.is_pinned()
    with pytest.raises(ValueError, match="16-byte aligned"):
        sgl_kernel.shadowkv_resolve_mapped_host_region(misaligned, device="cuda")


def test_shadowkv_mapped_host_bounds_failure_zeroes_complete_rows_and_guards():
    plan, temporal_values, compatibility, _, _ = _device_placement_arguments(
        "all-miss", parallel=True
    )
    host_values, compatibility = _mapped_host_placement_inputs(plan, compatibility)
    mapped = sgl_kernel.shadowkv_resolve_mapped_host_region(
        host_values,
        device="cuda",
    )
    guard_elements = 256
    output_elements = compatibility.numel()
    guarded = torch.full(
        (output_elements + 2 * guard_elements,),
        59,
        dtype=torch.bfloat16,
        device="cuda",
    )
    output = guarded.narrow(0, guard_elements, output_elements).view_as(compatibility)

    sgl_kernel.shadowkv_place_device_mapped_host(
        plan.component_kinds,
        plan.source_slots,
        plan.destination_slots,
        plan.miss_ordinals,
        plan.selected_chunk_ids,
        plan.plan_slots,
        plan.error_codes,
        temporal_values,
        compatibility[0],
        plan.value_miss_chunk_ids,
        plan.value_miss_lengths,
        torch.tensor([17], dtype=torch.int64, device="cuda"),
        torch.ones((1,), dtype=torch.uint8, device="cuda"),
        mapped,
        output,
        prompt_tokens=4 * 8,
        expected_generation=17,
        plan_capacity=2,
    )

    assert plan.error_codes.cpu().tolist() == [8, 8]
    assert not torch.count_nonzero(output).item()
    assert torch.equal(
        guarded[:guard_elements], torch.full_like(guarded[:guard_elements], 59)
    )
    assert torch.equal(
        guarded[-guard_elements:], torch.full_like(guarded[-guard_elements:], 59)
    )


def test_shadowkv_place_device_zeroes_invalid_rows_and_destinations():
    plan, temporal_values, compatibility, exact, exact_lengths = (
        _device_placement_arguments("all-hit")
    )
    plan.error_codes[0] = 3
    plan.destination_slots[0, 1, 0] = -1
    output = torch.full_like(compatibility, 17)

    sgl_kernel.shadowkv_place_device(
        plan.component_kinds,
        plan.source_slots,
        plan.destination_slots,
        plan.plan_slots,
        plan.error_codes,
        temporal_values,
        compatibility,
        output,
        plan_capacity=2,
    )

    assert not torch.count_nonzero(output[:, 0]).item()
    assert not torch.count_nonzero(output[0, 1, 0]).item()
    assert torch.count_nonzero(output[1, 1, 0]).item() == 8 * 128
    assert plan.error_codes.cpu().tolist() == [3, 8]

    temporal_ids = torch.full((1, 2, 2, 3), 23, dtype=torch.int32, device="cuda")
    temporal_before = temporal_ids.clone()
    published_values = torch.full_like(temporal_values, 29)
    values_before = published_values.clone()
    publication_generations = torch.full(
        (2, 1, 2, 2, 3), 31, dtype=torch.int64, device="cuda"
    )
    published_validity = torch.ones((2, 1, 2, 2, 3), dtype=torch.uint8, device="cuda")
    sgl_kernel.shadowkv_publish_device(
        plan.selected_chunk_ids,
        plan.selected_lengths,
        exact,
        exact_lengths,
        plan.row_indices,
        plan.row_generations,
        plan.error_codes,
        output,
        torch.tensor([7], dtype=torch.int64, device="cuda"),
        torch.tensor([3], dtype=torch.int64, device="cuda"),
        temporal_ids,
        published_values,
        publication_generations,
        published_validity,
    )
    assert torch.equal(temporal_ids[0, 1, 0], temporal_before[0, 1, 0])
    assert torch.equal(temporal_ids[0, 1, 1], temporal_before[0, 1, 1])
    assert torch.equal(published_values[:, 0, 1, 0], values_before[:, 0, 1, 0])
    assert torch.equal(published_values[:, 0, 1, 1], values_before[:, 0, 1, 1])


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
    parallel = sgl_kernel.shadowkv_plan_device_v2(
        *arguments, plan_capacity=plan_capacity
    )
    expected_value_misses = _value_miss_descriptor_reference(arguments, expected)

    assert not actual.error_codes.any().item()
    assert torch.equal(actual.component_kinds, expected[0])
    assert torch.equal(actual.source_slots, expected[1])
    assert torch.equal(actual.destination_slots, expected[2])
    assert torch.equal(actual.miss_ordinals, expected[3])
    assert torch.equal(actual.counts, expected[4])
    assert torch.equal(parallel.component_kinds, expected[0])
    assert torch.equal(parallel.source_slots, expected[1])
    assert torch.equal(parallel.destination_slots, expected[2])
    assert torch.equal(parallel.miss_ordinals, expected[3])
    assert torch.equal(parallel.counts, expected[4])
    assert torch.equal(parallel.value_miss_chunk_ids, expected_value_misses[0])
    assert torch.equal(parallel.value_miss_lengths, expected_value_misses[1])


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
    parallel = sgl_kernel.shadowkv_plan_device_v2(*arguments, plan_capacity=2)

    assert plan.error_codes.cpu().tolist() == [error_code]
    assert torch.equal(plan.component_kinds, torch.full_like(plan.component_kinds, -1))
    assert torch.equal(plan.source_slots, torch.full_like(plan.source_slots, -1))
    assert torch.equal(
        plan.destination_slots, torch.full_like(plan.destination_slots, -1)
    )
    assert torch.equal(plan.miss_ordinals, torch.full_like(plan.miss_ordinals, -1))
    assert torch.equal(plan.counts, torch.zeros_like(plan.counts))
    assert parallel.error_codes.cpu().tolist() == [error_code]
    assert torch.equal(
        parallel.component_kinds, torch.full_like(parallel.component_kinds, -1)
    )
    assert torch.equal(
        parallel.value_miss_chunk_ids,
        torch.full_like(parallel.value_miss_chunk_ids, -1),
    )
    assert torch.equal(
        parallel.value_miss_lengths,
        torch.zeros_like(parallel.value_miss_lengths),
    )


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
    outputs = sgl_kernel.ShadowKVDevicePlanOutputs(
        component_kinds=kinds,
        source_slots=sources,
        destination_slots=destinations,
        miss_ordinals=misses,
        counts=counts,
        error_codes=errors,
    )
    plan = sgl_kernel.shadowkv_plan_device(*arguments, plan_capacity=2, out=outputs)

    assert plan.component_kinds.data_ptr() == kinds.data_ptr()
    assert plan.error_codes.data_ptr() == errors.data_ptr()
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

    value_id_storage, value_ids = guarded((rows, selected_capacity), torch.int32, 1002)
    value_length_storage, value_lengths = guarded((rows,), torch.int32, 1003)
    parallel_outputs = sgl_kernel.ShadowKVDevicePlanV2Outputs(
        component_kinds=kinds,
        source_slots=sources,
        destination_slots=destinations,
        miss_ordinals=misses,
        counts=counts,
        error_codes=errors,
        value_miss_chunk_ids=value_ids,
        value_miss_lengths=value_lengths,
    )
    parallel = sgl_kernel.shadowkv_plan_device_v2(
        *arguments,
        plan_capacity=2,
        out=parallel_outputs,
    )
    expected_value_misses = _value_miss_descriptor_reference(arguments, expected)
    assert parallel.value_miss_chunk_ids.data_ptr() == value_ids.data_ptr()
    assert parallel.value_miss_lengths.data_ptr() == value_lengths.data_ptr()
    assert torch.equal(value_ids, expected_value_misses[0])
    assert torch.equal(value_lengths, expected_value_misses[1])
    for storage, fill in (
        (kind_storage, 99),
        (source_storage, 997),
        (destination_storage, 998),
        (miss_storage, 999),
        (count_storage, 1000),
        (error_storage, 1001),
        (value_id_storage, 1002),
        (value_length_storage, 1003),
    ):
        assert torch.equal(storage[:guard], torch.full_like(storage[:guard], fill))
        assert torch.equal(storage[-guard:], torch.full_like(storage[-guard:], fill))

    with pytest.raises(ValueError, match="out.error_codes must have shape"):
        sgl_kernel.shadowkv_plan_device(
            *arguments,
            plan_capacity=2,
            out=sgl_kernel.ShadowKVDevicePlanOutputs(
                component_kinds=kinds,
                source_slots=sources,
                destination_slots=destinations,
                miss_ordinals=misses,
                counts=counts,
                error_codes=torch.empty((rows + 1,), dtype=torch.int32, device="cuda"),
            ),
        )


def test_shadowkv_plan_device_wrapper_has_no_error_materialization():
    import inspect

    source = inspect.getsource(sgl_kernel.shadowkv_plan_device)
    assert ".cpu(" not in source
    assert ".item(" not in source
    parallel_source = inspect.getsource(sgl_kernel.shadowkv_plan_device_v2)
    assert ".cpu(" not in parallel_source
    assert ".item(" not in parallel_source


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


def test_public_dispatch_matches_legacy_compatibility_wrappers():
    u = torch.zeros((8, 160), dtype=torch.bfloat16, device="cuda")
    positions = torch.tensor([[0, 7]], dtype=torch.int64, device="cuda")
    reconstruct_sv = torch.zeros((1, 160, 128), dtype=torch.bfloat16, device="cuda")
    rope_sv = torch.zeros((1, 160, 64), dtype=torch.bfloat16, device="cuda")
    inverse = torch.ones((32,), dtype=torch.float32, device="cuda")
    torch.testing.assert_close(
        shadowkv_api.reconstruct(
            u,
            reconstruct_sv,
            positions,
            implementation="shadowkv.reconstruct.generic-aot.v1",
        ),
        sgl_kernel.shadowkv_reconstruct(u, reconstruct_sv, positions),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        shadowkv_api.reconstruct_rope(
            u,
            rope_sv,
            positions,
            inverse,
            implementation="shadowkv.reconstruct-rope.generic-aot.v1",
        ),
        sgl_kernel.shadowkv_reconstruct_rope(u, rope_sv, positions, inverse),
        rtol=0,
        atol=0,
    )

    planner = (
        torch.tensor([[1, 4]], dtype=torch.int64, device="cuda"),
        torch.tensor([2], dtype=torch.int32, device="cuda"),
        torch.tensor([[4, 5]], dtype=torch.int64, device="cuda"),
        torch.tensor([2], dtype=torch.int32, device="cuda"),
        torch.tensor([[5]], dtype=torch.int64, device="cuda"),
        torch.tensor([1], dtype=torch.int32, device="cuda"),
        torch.tensor([3], dtype=torch.int64, device="cuda"),
        torch.tensor([3], dtype=torch.int64, device="cuda"),
    )
    public_plan = shadowkv_api.plan_reuse(
        *planner,
        max_reuse_chunks=2,
        chunk_size=8,
        implementation="shadowkv.plan-reuse.generic-aot.v1",
    )
    legacy_plan = sgl_kernel.shadowkv_plan_reuse(
        *planner, max_reuse_chunks=2, chunk_size=8
    )
    assert torch.equal(public_plan.plan, legacy_plan.plan)
    assert torch.equal(
        public_plan.deduplicated_exact_chunks,
        legacy_plan.deduplicated_exact_chunks,
    )
    assert torch.equal(public_plan.counts, legacy_plan.counts)

    query = torch.zeros((1, 4, 64), dtype=torch.bfloat16, device="cuda")
    keys = torch.zeros((1, 1, 7, 64), dtype=torch.bfloat16, device="cuda")
    values = torch.zeros_like(keys)
    lengths = torch.tensor([7], dtype=torch.int32, device="cuda")
    torch.testing.assert_close(
        shadowkv_api.packed_gqa(
            query,
            keys,
            values,
            lengths,
            implementation="shadowkv.packed-gqa.generic-aot.v1",
        ),
        sgl_kernel.shadowkv_packed_gqa(query, keys, values, lengths),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "legacy,new,message",
    [
        (
            lambda: sgl_kernel.shadowkv_reconstruct(
                torch.zeros((8, 96), dtype=torch.bfloat16, device="cuda"),
                torch.zeros((1, 96, 128), dtype=torch.bfloat16, device="cuda"),
                torch.zeros((1, 1), dtype=torch.int64, device="cuda"),
            ),
            lambda: shadowkv_api.reconstruct(
                torch.zeros((8, 96), dtype=torch.bfloat16, device="cuda"),
                torch.zeros((1, 96, 128), dtype=torch.bfloat16, device="cuda"),
                torch.zeros((1, 1), dtype=torch.int64, device="cuda"),
                implementation="shadowkv.reconstruct.generic-aot.v1",
            ),
            "rank must be one of",
        ),
        (
            lambda: sgl_kernel.shadowkv_packed_gqa(
                torch.zeros((1, 3, 64), dtype=torch.bfloat16, device="cuda"),
                torch.zeros((1, 2, 7, 64), dtype=torch.bfloat16, device="cuda"),
                torch.zeros((1, 2, 7, 64), dtype=torch.bfloat16, device="cuda"),
                torch.ones((1,), dtype=torch.int32, device="cuda"),
            ),
            lambda: shadowkv_api.packed_gqa(
                torch.zeros((1, 3, 64), dtype=torch.bfloat16, device="cuda"),
                torch.zeros((1, 2, 7, 64), dtype=torch.bfloat16, device="cuda"),
                torch.zeros((1, 2, 7, 64), dtype=torch.bfloat16, device="cuda"),
                torch.ones((1,), dtype=torch.int32, device="cuda"),
                implementation="shadowkv.packed-gqa.generic-aot.v1",
            ),
            "query heads must be divisible",
        ),
    ],
)
def test_public_dispatch_matches_legacy_error_boundary(legacy, new, message):
    with pytest.raises(ValueError, match=message) as legacy_error:
        legacy()
    with pytest.raises(type(legacy_error.value), match=message) as public_error:
        new()
    assert str(public_error.value) == str(legacy_error.value)
