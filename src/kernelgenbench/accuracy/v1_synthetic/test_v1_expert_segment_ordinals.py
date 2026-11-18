"""Accuracy and performance for v1_expert_segment_ordinals.

Official entry: ``deepspeed.moe.ep_tp_dispatch.assignment_ordinals_by_expert``
(DeepSpeed 8af59b2e7a91c227081617240e98aa282eadc5f9, ep_tp_dispatch.py
L43-L54).  Ordinals restart at 0 for each run of equal adjacent expert ids and
are always int64.  Empty int64 input is returned as the same object; every
non-empty input produces fresh storage.  A 2-D input is rejected by the source.
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


# (expert id values, dtype, strided view, device)
COMBOS = [
    ([], torch.int32, False, "cuda"),
    ([], torch.int64, False, "cuda"),
    ([0], torch.int32, False, "cuda"),
    ([0], torch.int64, False, "cuda"),
    ([0, 0, 2, 2, 2, 1, 1, -1, -1, 0], torch.int32, False, "cuda"),
    ([0, 0, 2, 2, 2, 1, 1, -1, -1, 0], torch.int64, False, "cuda"),
    ([3, 3, 3, 3], torch.int32, True, "cpu"),
    ([3, 3, 3, 3], torch.int64, True, "cpu"),
    ([1, 2, 3, 4, 5, 6, 7, 8, 9], torch.int64, False, "cpu"),
]


def input_build(config):
    values, dtype, strided, device = config
    tensor = torch.tensor(values, dtype=dtype)
    if strided:
        tensor = tensor.repeat_interleave(2)[::2]
    return {"expert_indices": tensor}


def compute_ref(expert_indices):
    return kernelgenbench.baseline.v1_expert_segment_ordinals(expert_indices)


def compute_res(expert_indices):
    return kernelgenbench.solution.v1_expert_segment_ordinals(expert_indices)


def compare(ref_out, res_out):
    assert isinstance(ref_out, torch.Tensor) and isinstance(res_out, torch.Tensor)
    assert ref_out.shape == res_out.shape
    assert ref_out.dtype == res_out.dtype == torch.int64
    assert torch.equal(ref_out, res_out)


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_expert_segment_ordinals")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_expert_segment_ordinals(config):
    values, dtype, strided, device = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, device)
    res_inputs = fresh(inputs, device)
    before = res_inputs["expert_indices"].clone()
    if strided:
        assert res_inputs["expert_indices"].stride(0) == 2
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out)
    assert torch.equal(res_inputs["expert_indices"], before)
    # Empty int64 ``.to(torch.long)`` is a no-op in the source; empty int32 is cast.
    if not values:
        assert (ref_out is ref_inputs["expert_indices"]) == (dtype == torch.int64)
        assert (res_out is res_inputs["expert_indices"]) == (dtype == torch.int64)
    else:
        assert ref_out is not ref_inputs["expert_indices"]
        assert res_out is not res_inputs["expert_indices"]

    # The source contract is one-dimensional and fails on 2-D input.
    if config is COMBOS[0]:
        bad = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64, device=device)
        for fn in (compute_ref, compute_res):
            try:
                fn(expert_indices=bad)
            except RuntimeError:
                pass
            else:
                raise AssertionError("2-D expert_indices must fail")

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return timings(ref_time, res_time)
