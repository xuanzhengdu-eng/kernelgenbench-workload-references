"""Accuracy and performance for SGLang MoE expert alignment.

The check independently walks the ``-1`` invalid bucket and experts
``0..num_experts-1``.  For each bucket it requires the exact routed positions,
the block-aligned padding sentinel and the matching block expert id; the
returned active length must equal the padded prefix.  Expert ids past the
active prefix are left uninitialized by the upstream kernel and are not
asserted.  The legacy Megatron-fork alignment branch has no NVIDIA Megatron
entry at the pinned commit and is not registered.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.bias_activation_test_utils import fresh, measure
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (routing ids, experts, block size, ignore invalid, dtype)
COMBOS = [
    ([[0, -1], [2, 1], [0, -1]], 3, 4, False, "int32"),
    ([[0, -1], [2, 1], [0, -1]], 3, 4, True, "int64"),
    ([[-1, 1]], 5, 8, False, "int32"),
]


def input_build(config):
    ids, experts, block, ignore, dtype = config
    routing_map = torch.tensor(ids, dtype=getattr(torch, dtype))
    return dict(
        routing_map=routing_map,
        block_size=block,
        num_local_experts=experts,
        valid_tokens=torch.tensor(routing_map.shape[0], dtype=torch.int32),
        ignore_invalid_experts=ignore,
    )


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_megatron_moe_align_graphable_sglang(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_megatron_moe_align_graphable_sglang(**inputs)


def check(inputs, result):
    assert len(result) == 3
    ids, experts, total = result
    routing_map = inputs["routing_map"]
    num_experts = inputs["num_local_experts"]
    block = inputs["block_size"]
    count = routing_map.numel()
    ignore = inputs["ignore_invalid_experts"]
    size = count * block if count < num_experts + 1 else count + (num_experts + 1) * (block - 1)
    size = (size + 3) // 4 * 4
    assert ids.shape == (size,)
    assert experts.shape == ((size + block - 1) // block,)
    assert total.shape == (1,)
    assert ids.dtype == experts.dtype == total.dtype == torch.int32

    flat = routing_map.flatten().tolist()
    cursor = 0
    for expert in range(-1, num_experts):
        if expert == -1 and ignore:
            continue
        positions = [index for index, value in enumerate(flat) if value == expert]
        number = len(positions)
        padded = (number + block - 1) // block * block
        expected = torch.tensor(positions, dtype=torch.int32, device=routing_map.device)
        assert torch.equal(ids[cursor:cursor + number].sort().values, expected)
        assert (ids[cursor + number:cursor + padded] == count).all()
        assert (experts[cursor // block:(cursor + padded) // block] == expert).all()
        cursor += padded
    assert int(total) == cursor
    assert (ids[cursor:] == count).all()


def compare(ref_out, res_out, inputs):
    check(inputs, ref_out)
    check(inputs, res_out)


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_megatron_moe_align_graphable_sglang")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_moe_align_graphable_sglang(config):
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
