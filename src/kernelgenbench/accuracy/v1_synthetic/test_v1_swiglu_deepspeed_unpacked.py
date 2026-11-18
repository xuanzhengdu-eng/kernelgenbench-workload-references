"""DeepSpeed unpacked SwiGLU forward; row 29, OP-d46764c4d7ca.

gate and up are separate same-shape tensors; the output is silu(gate) * up with
the same shape and dtype. This is split from the packed contracts because the
caller passes two tensors instead of one packed tensor. Cases cover three dtypes
and 1-D/2-D/3-D shapes. The solution binds swiglu, so the Triton
_swiglu_fwd_kernel runs.
"""
import pytest
import torch
import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.activation_rows22_31_checks import (
    config_dtype, fresh, measure, timings,
)
from sandbox.verifier.test_parametrize import label, parametrize


# (shape, dtype): gate and up have this shape.
COMBOS = [
    (shape, dtype)
    for shape in [(2, 16, 64), (4, 128), (1, 32)]
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
]


def input_build(config):
    """gate and up: identical shape and dtype, independent values."""
    shape, dtype = config
    rng = torch.Generator(device="cpu").manual_seed(2903)
    return {
        "gate": torch.randn(shape, dtype=dtype, generator=rng),
        "up": torch.randn(shape, dtype=dtype, generator=rng),
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_swiglu_deepspeed_unpacked(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_swiglu_deepspeed_unpacked(**inputs)


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


@label("v1_swiglu_deepspeed_unpacked")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_swiglu_deepspeed_unpacked(config):
    shape, dtype = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    assert ref_out.shape == shape
    for side in (ref_inputs, res_inputs):
        torch.testing.assert_close(side["gate"].cpu(), inputs["gate"], rtol=0, atol=0)
        torch.testing.assert_close(side["up"].cpu(), inputs["up"], rtol=0, atol=0)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return timings(ref_time, res_time)
