"""Megatron SwiGLUFunction backward; row 28, OP-bb5f17bc3006.

Explicit saved-input contract: returns (grad, None, None). When
fp8_input_store=True the saved tensor is FP8 E4M3 and original_dtype is used to
restore it before the elementwise backward. This split covers the autograd
wrapper's three-output contract; the fork-only clamp_value mode has no fixed
upstream counterpart and is removed. Cases cover three dtypes and both storage
modes. The solution builds the official ctx and calls SwiGLUFunction.backward.
"""
import pytest
import torch
import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.activation_rows22_31_checks import (
    config_dtype, fresh, measure, timings,
)
from sandbox.verifier.test_parametrize import label, parametrize


# (y_shape, dtype, fp8_input_store); even widths only for this wrapper test.
COMBOS = [
    (shape, dtype, fp8)
    for shape in [(2, 16, 128), (4, 64)]
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
    for fp8 in (False, True)
]


def input_build(config):
    """y [..., 2H] saved input; g [..., H]; FP8 mode stores y as e4m3fn."""
    shape, dtype, fp8 = config
    out_w = shape[-1] // 2
    rng = torch.Generator(device="cpu").manual_seed(2804)
    y = torch.randn(shape, dtype=dtype, generator=rng)
    g = torch.randn(shape[:-1] + (out_w,), dtype=dtype, generator=rng)
    saved = y.to(torch.float8_e4m3fn) if fp8 else y
    return {
        "g": g,
        "saved_input": saved,
        "original_dtype": dtype,
        "fp8_input_store": fp8,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_swiglu_back_megatron_autograd(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_swiglu_back_megatron_autograd(**inputs)


def compare(ref_out, res_out, dtype):
    assert isinstance(ref_out, tuple) and len(ref_out) == 3
    assert isinstance(res_out, tuple) and len(res_out) == 3
    assert res_out[1] is None and res_out[2] is None
    assert res_out[0].shape == ref_out[0].shape
    assert res_out[0].dtype == ref_out[0].dtype == dtype
    rtol, atol = {
        torch.float32: (4e-6, 2e-6),
        torch.float16: (3e-3, 2e-3),
        torch.bfloat16: (2e-2, 1.2e-2),
    }[dtype]
    torch.testing.assert_close(res_out[0], ref_out[0], rtol=rtol, atol=atol)


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_swiglu_back_megatron_autograd")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_swiglu_back_megatron_autograd(config):
    shape, dtype, fp8 = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    assert ref_out[0].shape == shape
    for side in (ref_inputs, res_inputs):
        torch.testing.assert_close(side["g"].cpu(), inputs["g"], rtol=0, atol=0)
        assert torch.equal(side["saved_input"].view(torch.uint8).cpu(),
                           inputs["saved_input"].view(torch.uint8))

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return timings(ref_time, res_time)
