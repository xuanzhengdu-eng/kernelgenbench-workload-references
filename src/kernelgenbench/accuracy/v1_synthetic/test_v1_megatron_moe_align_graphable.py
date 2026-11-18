"""Accuracy and performance for vLLM MoE expert alignment.

Each case supplies a literal routing map and, optionally, an expert map.  The
check walks the expected buckets independently of the candidate: every valid
route must appear once, each block must carry its expert id, padding must use
``tokens*topk``, the returned active length must match, and ``scatter_idx`` must
be the masked routed-row identity.  Routing map and ``valid_tokens`` must not
be mutated.  The legacy Megatron-fork ``moe_align_graphable`` branch has no
NVIDIA Megatron entry at the pinned commit and is not registered.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.bias_activation_test_utils import fresh, measure
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (routing ids, experts, block size, expert map, ignore invalid, pad, scatter, dtype)
COMBOS = [
    ([[0, 2], [4, -1], [1, 3], [5, 0]], 5, 4, None, False, False, True, "int32"),
    ([[0, 2], [4, -1], [1, 3], [5, 0]], 5, 4, [1, -1, 0, -1, 2], True, True, True, "int64"),
    ([[0, 2], [4, -1], [1, 3], [5, 0]], 5, 4, [1, -1, 0, -1, 2], False, False, False, "int32"),
    ([[2, 0]], 6, 8, [2, -1, 0, -1, -1, 1], True, False, True, "int32"),
]


def input_build(config):
    ids, experts, block, mapping, ignore, pad, scatter, dtype = config
    routing_map = torch.tensor(ids, dtype=getattr(torch, dtype))
    return dict(
        routing_map=routing_map,
        block_size=block,
        num_local_experts=experts,
        valid_tokens=torch.tensor(routing_map.shape[0], dtype=torch.int32),
        expert_map=None if mapping is None else torch.tensor(mapping, dtype=torch.int32),
        ignore_invalid_experts=ignore,
        pad_sorted_ids=pad,
        return_scatter_idx=scatter,
    )


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_megatron_moe_align_graphable(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_megatron_moe_align_graphable(**inputs)


def check(inputs, result):
    ids, experts, total = result[:3]
    routing_map = inputs["routing_map"]
    num_experts = inputs["num_local_experts"]
    block = inputs["block_size"]
    count = routing_map.numel()
    mapping = inputs["expert_map"]
    ignore = inputs["ignore_invalid_experts"]
    size = count + num_experts * (block - 1)
    if inputs["pad_sorted_ids"]:
        size = (size + block - 1) // block * block
    if count < num_experts:
        size = min(count * block, size)
    assert len(result) == (4 if inputs["return_scatter_idx"] else 3)
    assert ids.shape == (size,)
    assert experts.shape == ((size + block - 1) // block,)
    assert total.shape == (1,)
    assert ids.dtype == experts.dtype == total.dtype == torch.int32

    flat = routing_map.flatten()
    groups = {index: [] for index in range(num_experts)}
    valid = []
    for index, value in enumerate(flat.tolist()):
        in_range = 0 <= value < num_experts
        group = int(mapping[value]) if in_range and mapping is not None and ignore else value
        in_range = in_range and 0 <= group < num_experts
        valid.append(in_range)
        if in_range:
            groups[group].append(index)
    cursor = 0
    for expert in range(num_experts):
        expected = torch.tensor(groups[expert], device=routing_map.device, dtype=torch.int32)
        number = len(groups[expert])
        padded = (number + block - 1) // block * block
        assert torch.equal(ids[cursor:cursor + number].sort().values, expected.sort().values)
        assert (ids[cursor + number:cursor + padded] == count).all()
        mapped = int(mapping[expert]) if mapping is not None and not ignore else expert
        assert (experts[cursor // block:(cursor + padded) // block] == mapped).all()
        cursor += padded
    tail = int(mapping[-1]) if mapping is not None and not ignore else -1
    assert int(total) == cursor
    assert (ids[cursor:] == count).all()
    assert (experts[(cursor + block - 1) // block:] == tail).all()
    if inputs["return_scatter_idx"]:
        expected = torch.tensor([index if ok else -1 for index, ok in enumerate(valid)],
                                dtype=torch.int32, device=routing_map.device).reshape(-1, 1)
        assert torch.equal(result[3], expected)


def compare(ref_out, res_out, inputs):
    check(inputs, ref_out)
    check(inputs, res_out)


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_megatron_moe_align_graphable")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_moe_align_graphable(config):
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    routing_before = ref_inputs["routing_map"].clone()
    valid_before = ref_inputs["valid_tokens"].clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, ref_inputs)

    assert torch.equal(res_inputs["routing_map"], routing_before)
    assert torch.equal(res_inputs["valid_tokens"], valid_before)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
