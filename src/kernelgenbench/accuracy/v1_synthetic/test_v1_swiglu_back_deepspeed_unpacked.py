"""DeepSpeed unpacked SwiGLU backward; row 28, OP-bb5f17bc3006.

gate, up and g are three separate tensors of the same shape; the result is the
two-tuple (grad_gate, grad_up). This is a distinct caller-visible contract from
the packed entries (separate inputs, tuple output). Cases cover three dtypes and
1-D/2-D/3-D shapes. The solution autograds the upstream swiglu Function.
"""
import pytest
import torch
import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.activation_rows22_31_checks import (
    config_dtype, fresh, measure, timings,
)
from sandbox.verifier.test_parametrize import label, parametrize


# (shape, dtype): gate, up and g all have this shape.
COMBOS = [
    (shape, dtype)
    for shape in [(2, 16, 64), (4, 128), (1, 32)]
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
]


def input_build(config):
    """gate, up and g: identical shape and dtype, independent values."""
    shape, dtype = config
    rng = torch.Generator(device="cpu").manual_seed(2803)
    gate = torch.randn(shape, dtype=dtype, generator=rng)
    up = torch.randn(shape, dtype=dtype, generator=rng)
    g = torch.randn(shape, dtype=dtype, generator=rng)
    return {"g": g, "gate": gate, "up": up}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_swiglu_back_deepspeed_unpacked(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_swiglu_back_deepspeed_unpacked(**inputs)


def compare(ref_out, res_out, dtype):
    assert isinstance(ref_out, tuple) and len(ref_out) == 2
    assert isinstance(res_out, tuple) and len(res_out) == 2
    for ref_item, res_item in zip(ref_out, res_out):
        assert isinstance(res_item, torch.Tensor)
        assert res_item.shape == ref_item.shape
        assert res_item.dtype == ref_item.dtype == dtype
    rtol, atol = {
        torch.float32: (4e-6, 2e-6),
        torch.float16: (3e-3, 2e-3),
        torch.bfloat16: (2e-2, 1.2e-2),
    }[dtype]
    torch.testing.assert_close(res_out[0], ref_out[0], rtol=rtol, atol=atol)
    torch.testing.assert_close(res_out[1], ref_out[1], rtol=rtol, atol=atol)


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_swiglu_back_deepspeed_unpacked")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_swiglu_back_deepspeed_unpacked(config):
    shape, dtype = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    for side in (ref_inputs, res_inputs):
        for name in ("g", "gate", "up"):
            torch.testing.assert_close(side[name].cpu(), inputs[name], rtol=0, atol=0)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return timings(ref_time, res_time)
