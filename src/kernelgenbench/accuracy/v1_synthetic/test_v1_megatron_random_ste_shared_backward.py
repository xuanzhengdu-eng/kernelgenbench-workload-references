"""Accuracy and performance for v1_megatron_random_ste_shared_backward.

Official entry: ``RandomSTEShared.backward`` (Megatron-LM
266f1c97dca477ca8b92c16087506da2000b0b84, moe_utils.py L1245-L1278).
The forward takes ``(logits, std, layer_number)``, so the backward returns a
three-tuple: the unchanged gradient object followed by ``None`` for ``std`` and
``layer_number``.
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
    generator = torch.Generator(device="cpu").manual_seed(193)
    grad_output = torch.randn(shape, dtype=dtype, generator=generator)
    return {"grad_output": grad_output}


def compute_ref(grad_output):
    return kernelgenbench.baseline.v1_megatron_random_ste_shared_backward(grad_output)


def compute_res(grad_output):
    return kernelgenbench.solution.v1_megatron_random_ste_shared_backward(grad_output)


def compare(ref_out, res_out, dtype):
    assert isinstance(ref_out, tuple) and isinstance(res_out, tuple)
    assert len(ref_out) == len(res_out) == 3
    assert isinstance(ref_out[0], torch.Tensor) and isinstance(res_out[0], torch.Tensor)
    assert ref_out[0].shape == res_out[0].shape
    assert ref_out[0].dtype == res_out[0].dtype == dtype
    assert torch.equal(ref_out[0], res_out[0])
    assert ref_out[1] is None and ref_out[2] is None
    assert res_out[1] is None and res_out[2] is None


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_megatron_random_ste_shared_backward")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_random_ste_shared_backward(config):
    shape, dtype = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)
    # The first slot must be the incoming gradient object, not a clone.
    assert ref_out[0] is ref_inputs["grad_output"]
    assert res_out[0] is res_inputs["grad_output"]
    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return timings(ref_time, res_time)
