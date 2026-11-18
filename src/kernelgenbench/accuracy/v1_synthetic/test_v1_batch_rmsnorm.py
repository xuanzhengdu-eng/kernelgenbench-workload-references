"""Accuracy and performance for v1_batch_rmsnorm (row 63).

Contract: Megatron batch-invariant affine RMSNorm forward, zero_centered_gamma
False.  Statistics are fp32 (mean of squares along the last dimension); the
affine weight is applied in fp32 and the output is cast back to the input dtype.
Official: https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/custom_layers/batch_invariant_kernels.py#L877-L945
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows62_71_checks import (
    fresh, measure, result, tolerance, unchanged,
)
from sandbox.verifier.test_parametrize import label, parametrize

# (B, T, H) shape, input dtype and whether the weight is kept in fp32.
COMBOS = [
    (shape, dtype, weight_fp32)
    for shape in [(2, 16, 128), (2, 128, 1024), (8, 4096)]
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
    for weight_fp32 in (False, True)
]


def input_build(config):
    shape, dtype, weight_fp32 = config
    generator = torch.Generator(device="cpu").manual_seed(6301)
    x = torch.randn(shape, dtype=dtype, generator=generator)
    weight = torch.randn((shape[-1],), dtype=torch.float32 if weight_fp32 else dtype,
                         generator=generator)
    return {"x": x, "weight": weight, "eps": 1e-5}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_batch_rmsnorm(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_batch_rmsnorm(**inputs)


def compare(ref_out, res_out, dtype):
    assert isinstance(res_out, torch.Tensor) and isinstance(ref_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == dtype
    rtol, atol = tolerance(dtype)
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)


@label("v1_batch_rmsnorm")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_batch_rmsnorm(config):
    shape, dtype, weight_fp32 = config
    inputs = input_build(config)
    before = {k: v.clone() for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    assert ref_inputs["x"].dtype == dtype
    assert ref_inputs["weight"].dtype == (torch.float32 if weight_fp32 else dtype)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)
    unchanged(ref_inputs, before)
    unchanged(res_inputs, before)
    assert not torch._C._is_alias_of(res_out, res_inputs["x"])

    ref_time = measure(lambda: compute_ref(**ref_inputs))
    res_time = measure(lambda: compute_res(**res_inputs))
    return result(ref_time, res_time)
