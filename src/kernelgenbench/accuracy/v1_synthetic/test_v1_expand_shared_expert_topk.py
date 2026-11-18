"""Accuracy and performance for v1_expand_shared_expert_topk.

Official entry: ``sglang.srt.layers.moe.waterfill.expand_topk_with_shared_expert``
(SGLang 62384de1ae5e3bbd3159afd1639525b0bf39fbf9, waterfill.py L33-L68).
The function appends a local shared expert id, remaps routed ids to the
``old_experts_per_rank + 1`` layout, zeroes dropped routes, and uses the -1
sentinel plus zero weight for rows with no valid route.  The official function
is decorated with ``torch.compile(dynamic=True)``; the test calls it through
that compiled entry.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.moe_rows192_201_utils import (
    fresh,
    measure,
    timings,
)
from sandbox.verifier.test_parametrize import label, parametrize


# (tokens, topk, id dtype, weight dtype, source rank)
COMBOS = [
    (0, 3, torch.int32, torch.float32, 0),
    (5, 3, torch.int32, torch.float32, 1),
    (5, 3, torch.int64, torch.bfloat16, 0),
    (16, 8, torch.int64, torch.bfloat16, 1),
    (4, 2, torch.int32, torch.float16, 1),
]


def input_build(config):
    tokens, topk, id_dtype, weight_dtype, source_rank = config
    generator = torch.Generator(device="cpu").manual_seed(198)
    topk_ids = torch.randint(0, 8, (tokens, topk), dtype=id_dtype, generator=generator)
    topk_weights = torch.rand((tokens, topk), dtype=weight_dtype, generator=generator)
    if tokens:
        topk_ids[0, :] = -1
        if tokens > 1:
            topk_ids[1, 0] = -2
    return {
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "num_routed_experts": 8,
        "world_size": 2,
        "source_rank": source_rank,
        "shared_weight": 0.75,
    }


def compute_ref(topk_ids, topk_weights, num_routed_experts, world_size, source_rank, shared_weight):
    return kernelgenbench.baseline.v1_expand_shared_expert_topk(
        topk_ids, topk_weights, num_routed_experts, world_size, source_rank, shared_weight
    )


def compute_res(topk_ids, topk_weights, num_routed_experts, world_size, source_rank, shared_weight):
    return kernelgenbench.solution.v1_expand_shared_expert_topk(
        topk_ids, topk_weights, num_routed_experts, world_size, source_rank, shared_weight
    )


def compare(ref_out, res_out, topk):
    assert isinstance(ref_out, tuple) and isinstance(res_out, tuple)
    assert len(ref_out) == len(res_out) == 2
    for expected, actual in zip(ref_out, res_out):
        assert isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor)
        assert expected.shape == actual.shape
        assert expected.shape[-1] == topk + 1
        assert expected.dtype == actual.dtype
        assert torch.equal(expected, actual)


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_expand_shared_expert_topk")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_expand_shared_expert_topk(config):
    tokens, topk, id_dtype, weight_dtype, source_rank = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    before_ids = res_inputs["topk_ids"].clone()
    before_weights = res_inputs["topk_weights"].clone()
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, topk)
    assert torch.equal(res_inputs["topk_ids"], before_ids)
    assert torch.equal(res_inputs["topk_weights"], before_weights)

    # A row with no valid route gets the -1 sentinel and zero weight.
    if tokens:
        assert bool((res_out[0][0] == -1).all())
        assert bool((res_out[1][0] == 0).all())
    expected_shared = source_rank * (8 // 2 + 1) + (8 // 2)
    if tokens:
        assert torch.equal(res_out[0][1:, -1], torch.full_like(res_out[0][1:, -1], expected_shared))

    # Gradients for the appended weights: 1 on valid routes, 0 on dropped ones.
    for fn in (compute_ref, compute_res):
        weights = inputs["topk_weights"].clone().to(kernelgenbench.device).requires_grad_(True)
        out = fn(
            topk_ids=res_inputs["topk_ids"],
            topk_weights=weights,
            num_routed_experts=8,
            world_size=2,
            source_rank=source_rank,
            shared_weight=0.75,
        )
        out[1].sum().backward()
        expected_grad = torch.where(
            res_inputs["topk_ids"] >= 0,
            torch.ones_like(weights),
            torch.zeros_like(weights),
        )
        assert torch.equal(weights.grad, expected_grad)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return timings(ref_time, res_time)
