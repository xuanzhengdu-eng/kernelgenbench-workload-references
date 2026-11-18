"""Megatron packed SwiGLU forward; row 29, OP-d46764c4d7ca.

y [..., 2H] with an even last width splits with torch.chunk(2, -1); the output is
silu(first) * second and has width H. Cases cover three dtypes and even
widths. The solution binds megatron.core.fusions.fused_bias_swiglu.swiglu
with @jit_fuser intact.
"""
import pytest
import torch
import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.activation_rows22_31_checks import (
    config_dtype, fresh, measure, timings,
)
from sandbox.verifier.test_parametrize import label, parametrize


# (y_shape, dtype); y_shape[-1] is an even 2H.
COMBOS = [
    (shape, dtype)
    for shape in [(2, 16, 128), (2, 128, 1024), (1, 64)]
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
]


def input_build(config):
    """y: shape from config; the gate is the first torch.chunk half."""
    shape, dtype = config
    rng = torch.Generator(device="cpu").manual_seed(2901)
    return {"y": torch.randn(shape, dtype=dtype, generator=rng)}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_swiglu(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_swiglu(**inputs)


def compare(ref_out, res_out, dtype):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == dtype
    rtol, atol = {
        torch.float32: (4e-6, 2e-6),
        torch.float16: (3e-3, 2e-3),
        torch.bfloat16: (2e-2, 1.2e-2),
    }[dtype]
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_swiglu")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_swiglu(config):
    shape, dtype = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    expected = shape[:-1] + (shape[-1] // 2,)
    assert ref_out.shape == expected
    for side in (ref_inputs, res_inputs):
        torch.testing.assert_close(side["y"].cpu(), inputs["y"], rtol=0, atol=0)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return timings(ref_time, res_time)
