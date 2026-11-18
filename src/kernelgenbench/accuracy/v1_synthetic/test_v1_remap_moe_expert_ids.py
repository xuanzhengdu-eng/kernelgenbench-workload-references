"""Accuracy and performance for v1_remap_moe_expert_ids (vLLM Triton remap).

Official entry: vLLM ``remap_topk_to_local``, pinned commit
f358f0578a83f4e1fe6fc49a492d3ae9362de8e5, source lines 262-305.  The candidate
binding calls that Triton kernel with the requested ``out_dtype``; the Triton
launch is the real backend.

Input: ``topk_ids`` of several shapes (including empty and a multi-block
1025-element case), int32/int64, with negative invalid routes; ``expert_map``
of 17 entries with -1 non-local entries, int32/int64.  Output is a new tensor
of the same shape in int32/int64: invalid -> -1, valid -> expert_map[id].
Inputs are not modified.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic.rows202_211_test_utils import fresh, measure, result


SHAPES = [(0,), (1,), (2, 7), (33, 47), (1025,)]
COMBOS = [
    (shape, id_dtype, map_dtype, out_dtype)
    for shape in SHAPES
    for id_dtype in (torch.int32, torch.int64)
    for map_dtype in (torch.int32, torch.int64)
    for out_dtype in (torch.int32, torch.int64)
]

EXPERT_MAP = [3, -1, 0, 2, -1, 8, 4, 1, -1, 5, 7, -1, 6, 10, 9, -1, 11]


def input_build(config):
    """IDs range over [-3, 16]; the map has both local and -1 entries."""
    shape, id_dtype, map_dtype, out_dtype = config
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    topk_ids = torch.randint(-3, 17, shape, generator=generator, dtype=id_dtype)
    expert_map = torch.tensor(EXPERT_MAP, dtype=map_dtype)
    return {"topk_ids": topk_ids, "expert_map": expert_map, "out_dtype": out_dtype}


def compute_ref(topk_ids, expert_map, out_dtype):
    return kernelgenbench.baseline.v1_remap_moe_expert_ids(topk_ids, expert_map, out_dtype)


def compute_res(topk_ids, expert_map, out_dtype):
    return kernelgenbench.solution.v1_remap_moe_expert_ids(topk_ids, expert_map, out_dtype)


def compare(ref_out, res_out, dtype):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == dtype
    assert res_out.device == ref_out.device
    assert torch.equal(res_out, ref_out)


def compute_ref_performance(inputs):
    """Time the reference on independently rebuilt inputs."""
    return measure(compute_ref, lambda: fresh(inputs, kernelgenbench.device))


def compute_res_performance(inputs):
    """Time the candidate on independently rebuilt inputs."""
    return measure(compute_res, lambda: fresh(inputs, kernelgenbench.device))


@label("v1_remap_moe_expert_ids")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_remap_moe_expert_ids(config):
    shape, id_dtype, map_dtype, out_dtype = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    before = {k: v.clone() for k, v in res_inputs.items() if torch.is_tensor(v)}

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, out_dtype)

    # Independent mapping check, including negative sentinels.
    expected = torch.where(
        res_inputs["topk_ids"] >= 0,
        res_inputs["expert_map"][res_inputs["topk_ids"].clamp(min=0).long()].to(out_dtype),
        torch.full_like(res_out, -1),
    )
    assert torch.equal(res_out, expected), "global->local mapping contract"
    for key, value in before.items():
        assert torch.equal(res_inputs[key], value), f"{key} must not change"
        if res_inputs[key].numel():
            assert res_out.data_ptr() != res_inputs[key].data_ptr()

    ref_ms = compute_ref_performance(inputs)
    res_ms = compute_res_performance(inputs)
    return result(ref_ms, res_ms)
