"""Accuracy and performance for v1_megatron_inference_moe_permute.

Contract: ``megatron.core.inference.moe.permute.permute_tokens``. Tokens are
bucketed per local expert, each expert's block is padded up to ``alignment``,
and the four outputs are ``(permuted_hidden, permuted_probs, permutation_map,
inclusive_offsets)``. Padding entries of ``permutation_map`` are ``-1`` and the
payload of padding rows is uninitialized. The slot order inside one expert block
comes from atomic increments and is NOT part of the contract, so comparisons
canonicalize each block by source token index.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import moe_rows222_231_checks as checks


# (num_tokens, hidden, topk, local_expert_start, num_local_experts, alignment, dtype)
COMBOS = [
    (6, 8, 2, 1, 4, 6, torch.float32),
    (6, 16, 2, 1, 4, 3, torch.bfloat16),
    (5, 8, 2, 0, 3, 2, torch.float32),
    (5, 8, 2, 0, 4, 1, torch.bfloat16),
    (1, 1, 1, 0, 1, 1, torch.float32),
    (0, 4, 2, 0, 2, 3, torch.float32),
    (7, 12, 3, 4, 2, 4, torch.float16),
]


def input_build(config):
    num_tokens, hidden, topk, start, experts, alignment, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(223)
    hidden_states = torch.randn((num_tokens, hidden), dtype=dtype, generator=generator)
    probs = torch.rand((num_tokens, topk), dtype=torch.float32, generator=generator)
    # Deterministic global expert ids around ``start``. Unique (token, slot)
    # routes and an expert with zero traffic are both covered.
    routing_map = torch.empty((num_tokens, topk), dtype=torch.int32)
    for token in range(num_tokens):
        for slot in range(topk):
            routing_map[token, slot] = start + ((token * topk + slot) % (experts + 1))
    return {
        "hidden_states": hidden_states,
        "probs": probs,
        "routing_map": routing_map,
        "local_expert_start": start,
        "num_local_experts": experts,
        "alignment": alignment,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_megatron_inference_moe_permute(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_megatron_inference_moe_permute(**inputs)


def _canonical_blocks(result):
    """Sort each expert block's live rows by source token, keep block offsets."""
    hidden, probs, mapping, offsets = result
    blocks = []
    previous = 0
    for end in offsets.tolist():
        block_map = mapping[previous:end]
        live = torch.nonzero(block_map >= 0, as_tuple=False).flatten()
        order = live[torch.argsort(block_map[live])]
        blocks.append(
            (
                block_map[order],
                hidden[previous + order],
                probs[previous + order],
                int((block_map < 0).sum().item()),
            )
        )
        previous = end
    return blocks


def compare(ref_out, res_out, dtype=None):
    assert isinstance(ref_out, tuple) and isinstance(res_out, tuple)
    assert len(ref_out) == len(res_out) == 4
    ref_hidden, ref_probs, ref_map, ref_offsets = ref_out
    res_hidden, res_probs, res_map, res_offsets = res_out
    assert ref_hidden.shape == res_hidden.shape
    assert ref_hidden.dtype == res_hidden.dtype == dtype
    assert ref_probs.shape == res_probs.shape and ref_probs.dtype == res_probs.dtype
    assert ref_map.shape == res_map.shape and ref_map.dtype == res_map.dtype == torch.int32
    assert ref_offsets.shape == res_offsets.shape
    assert ref_offsets.dtype == res_offsets.dtype == torch.int32
    assert torch.equal(ref_offsets, res_offsets), "aligned inclusive offsets differ"
    for ref_block, res_block in zip(_canonical_blocks(ref_out), _canonical_blocks(res_out)):
        assert torch.equal(ref_block[0], res_block[0]), "live token sets per block differ"
        assert ref_block[3] == res_block[3], "padding counts per block differ"
        assert torch.equal(ref_block[1], res_block[1]), "permuted hidden payload differs"
        assert torch.equal(ref_block[2], res_block[2]), "permuted prob payload differs"


def compute_ref_performance(**inputs):
    return checks.measure(kernelgenbench.baseline.v1_megatron_inference_moe_permute, inputs, device=kernelgenbench.device)


def compute_res_performance(**inputs):
    return checks.measure(kernelgenbench.solution.v1_megatron_inference_moe_permute, inputs, device=kernelgenbench.device)


@label("v1_megatron_inference_moe_permute")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_inference_moe_permute(config):
    num_tokens, hidden, topk, start, experts, alignment, dtype = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    before = {
        name: value.cpu().clone()
        for name, value in res_inputs.items()
        if isinstance(value, torch.Tensor)
    }

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    # Contract: expected aligned offsets, live rows are the routed pairs, and
    # padding slots are -1 in every block (payload of padding rows is undefined).
    routes = res_inputs["routing_map"]
    counts = []
    for expert in range(experts):
        count = int((routes == start + expert).sum().item())
        counts.append(((count + alignment - 1) // alignment) * alignment if count else 0)
    expected_offsets = torch.tensor(counts, dtype=torch.int32, device=res_out[3].device).cumsum(0).to(torch.int32)
    assert torch.equal(res_out[3], expected_offsets), "offsets are not aligned prefix sums"
    assert res_out[2].shape == (num_tokens * min(topk, experts) + alignment * experts,)
    hidden_out, probs_out, mapping, _ = res_out
    previous = 0
    for expert, count in enumerate(counts):
        block = mapping[previous : previous + count]
        live = block >= 0
        assert int((~live).sum().item()) == count - int(
            (routes == start + expert).sum().item()
        )
        for position in torch.nonzero(live, as_tuple=False).flatten().tolist():
            token = int(mapping[previous + position])
            slots = torch.nonzero(routes[token] == start + expert, as_tuple=False).flatten()
            assert slots.numel() == 1, "test routes must be unique per (token, expert)"
            assert torch.equal(hidden_out[previous + position], res_inputs["hidden_states"][token])
            assert probs_out[previous + position] == res_inputs["probs"][token, int(slots[0])]
            assert int(block[position]) == token
        previous += count
    for name, value in before.items():
        assert torch.equal(res_inputs[name].cpu(), value), f"{name} was modified"

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return checks.timings(ref_time, res_time)
