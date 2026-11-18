"""Accuracy and performance for v1_megatron_random_ste_backward.

Official entry: ``RandomSTE.backward`` (Megatron-LM
266f1c97dca477ca8b92c16087506da2000b0b84, moe_utils.py L1194-L1230).
The backward is the straight-through identity: it returns the same gradient
object, not a copy, and leaves shape/dtype/device/strides untouched.
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


COMBOS = [
    ((0, 7), torch.float32),
    ((3, 7), torch.float32),
    ((8, 16), torch.float16),
    ((2, 5, 9), torch.bfloat16),
]


def input_build(config):
    shape, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(192)
    grad_output = torch.randn(shape, dtype=dtype, generator=generator)
    return {"grad_output": grad_output}


def compute_ref(grad_output):
    return kernelgenbench.baseline.v1_megatron_random_ste_backward(grad_output)


def compute_res(grad_output):
    return kernelgenbench.solution.v1_megatron_random_ste_backward(grad_output)


def compare(ref_out, res_out, dtype):
    assert isinstance(ref_out, torch.Tensor) and isinstance(res_out, torch.Tensor)
    assert ref_out.shape == res_out.shape
    assert ref_out.dtype == res_out.dtype == dtype
    assert torch.equal(ref_out, res_out)


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_megatron_random_ste_backward")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_random_ste_backward(config):
    shape, dtype = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)
    # The defining contract: the returned object is the incoming gradient.
    assert ref_out is ref_inputs["grad_output"]
    assert res_out is res_inputs["grad_output"]
    for side in (ref_inputs, res_inputs):
        assert side["grad_output"].shape == shape
        assert side["grad_output"].dtype == dtype
    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return timings(ref_time, res_time)
