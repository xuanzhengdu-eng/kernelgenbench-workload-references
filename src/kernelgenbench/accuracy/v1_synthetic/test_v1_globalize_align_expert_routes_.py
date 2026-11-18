"""Accuracy and performance for v1_globalize_align_expert_routes_.

Official entry: ``vllm.model_executor.layers.fused_moe.
fused_globalize_align_block_size.fused_globalize_align_block_size``
(vLLM f358f0578a83f4e1fe6fc49a492d3ae9362de8e5, Python L10-L63 and CUDA
csrc/moe/fused_globalize_align_block_size.cu L28-L126).  The test runs the real
``torch.ops._moe_C`` kernel through that entry.  Valid local ids are globalized
in place; negative, out-of-range and beyond-psum rows become -1; every expert
gets padded blocks with the numel sentinel in unused slots.  The CUDA source
uses atomics, so within-expert order is compared as a set.
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


# (rows, topk, local experts, block size, valid rows, rank offset, global experts, scatter)
COMBOS = [
    (0, 2, 4, 4, 0, 2, 6, False),
    (1, 2, 4, 4, 1, 2, 6, True),
    (4, 3, 5, 4, 2, 3, 6, False),
    (16, 2, 4, 8, 16, 0, 4, True),
    (512, 4, 8, 16, 500, 3, 11, True),
]


def input_build(config):
    rows, topk, experts, block, valid_rows, offset, global_experts, scatter = config
    generator = torch.Generator(device="cpu").manual_seed(199)
    recv_topk_idx = torch.randint(
        -2, experts + 2, (rows, topk), dtype=torch.int64, generator=generator
    )
    return {
        "recv_topk_idx": recv_topk_idx,
        "psum_recv_per_rank": torch.tensor([0, valid_rows], dtype=torch.int32),
        "rank_expert_offset": offset,
        "global_num_experts": global_experts,
        "local_num_experts": experts,
        "block_size": block,
        "return_scatter_idx": scatter,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_globalize_align_expert_routes_(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_globalize_align_expert_routes_(**inputs)


def compare(ref_out, res_out, inputs):
    expected_count = 5 if inputs["return_scatter_idx"] else 4
    assert isinstance(ref_out, tuple) and isinstance(res_out, tuple)
    assert len(ref_out) == len(res_out) == expected_count
    for expected, actual in zip(ref_out, res_out):
        assert isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor)
        assert expected.shape == actual.shape
        assert expected.dtype == actual.dtype
        assert expected.device == actual.device
    assert res_out[0] is inputs["recv_topk_idx"]
    assert torch.equal(ref_out[0], res_out[0])
    assert torch.equal(ref_out[2], res_out[2])
    assert torch.equal(ref_out[3], res_out[3])
    if expected_count == 5:
        assert torch.equal(ref_out[4], res_out[4])

    block = inputs["block_size"]
    experts = inputs["local_num_experts"]
    expected_sorted, actual_sorted = ref_out[1], res_out[1]
    expert_ids = ref_out[2]
    for expert in range(experts):
        mask = (expert_ids == expert).repeat_interleave(block)[: expected_sorted.numel()]
        assert torch.equal(
            torch.sort(expected_sorted[mask]).values,
            torch.sort(actual_sorted[mask]).values,
        ), f"expert {expert} differs"
    sentinel_mask = (expert_ids < 0).repeat_interleave(block)[: expected_sorted.numel()]
    assert torch.equal(expected_sorted[sentinel_mask], actual_sorted[sentinel_mask])


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_globalize_align_expert_routes_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_globalize_align_expert_routes_(config):
    rows, topk, experts, block, valid_rows, offset, global_experts, scatter = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    psum_before = res_inputs["psum_recv_per_rank"].clone()
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, res_inputs)
    assert torch.equal(res_inputs["psum_recv_per_rank"], psum_before)
    assert res_out[0].shape == (rows, topk)
    assert res_out[3].shape == (1,)
    assert int(res_out[3].item()) % block == 0
    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return timings(ref_time, res_time)
