"""SGLang Conv3d: unfold+linear only for unit dilation, else dilated conv3d."""
import kernelgenbench
import pytest
import torch
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic.rows422_431_utils import (
    assert_no_alias, assert_unchanged, fresh, measure)

# (input shape, weight shape, stride, padding, dilation, groups, bias, linear, dtype)
COMBOS = [
    ((2, 2, 4, 6, 8), (3, 2, 2, 2, 2), 2, 0, 1, 1, True, True, torch.float32),
    ((1, 2, 5, 7, 5), (4, 2, 2, 2, 2), 2, 0, 1, 1, False, True, torch.float32),
    ((1, 4, 5, 6, 5), (6, 2, 3, 3, 3), 1, 1, 1, 2, True, True, torch.float32),
    ((1, 2, 6, 7, 6), (3, 2, 2, 2, 2), 1, 1, 2, 1, False, True, torch.float32),
    ((1, 1, 8, 8, 8), (1, 1, 2, 2, 2), 2, 0, 2, 1, True, True, torch.float32),
    ((2, 2, 4, 6, 8), (3, 2, 2, 2, 2), 2, 0, 1, 1, True, False, torch.float16),
    ((1, 2, 5, 7, 5), (4, 2, 2, 2, 2), 2, 0, 1, 1, False, True, torch.bfloat16),
]


def input_build(config):
    shape, weight_shape, stride, padding, dilation, groups, bias, linear, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(20261115)
    x = torch.randn(shape, dtype=dtype, generator=generator)
    weight = torch.randn(weight_shape, dtype=dtype, generator=generator) * 0.2
    bias_t = torch.randn((weight_shape[0],), dtype=dtype, generator=generator) * 0.1 if bias else None
    return {"x": x, "weight": weight, "bias": bias_t, "stride": (stride,) * 3,
            "padding": (padding,) * 3, "dilation": (dilation,) * 3, "groups": groups,
            "linear": linear}


def compute_ref(x, weight, bias, stride, padding, dilation, groups, linear=True):
    return kernelgenbench.baseline.v1_sglang_conv3d(x, weight, bias, stride, padding,
                                                    dilation, groups, linear)


def compute_res(x, weight, bias, stride, padding, dilation, groups, linear=True):
    return kernelgenbench.solution.v1_sglang_conv3d(x, weight, bias, stride, padding,
                                                    dilation, groups, linear)


def compare(ref_out, res_out, dtype):
    assert res_out.shape == ref_out.shape and res_out.dtype == ref_out.dtype == dtype
    assert res_out.stride() == ref_out.stride()
    rtol, atol = {torch.float32: (2e-5, 2e-5), torch.float16: (3e-2, 3e-2),
                  torch.bfloat16: (3e-2, 3e-2)}[dtype]
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_sglang_conv3d")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sglang_conv3d(config):
    shape, weight_shape, stride, padding, dilation, groups, bias, linear, dtype = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    saved = {name: value.clone() for name, value in inputs.items() if isinstance(value, torch.Tensor)}

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    admissible = (weight_shape[2:] == (stride,) * 3 and padding == 0
                  and dilation == 1 and groups == 1)
    # A single output channel makes the permuted linear view contiguous too.
    assert res_out.is_contiguous() == (not (linear and admissible)) or weight_shape[0] == 1
    # With a non-unit dilation SGLang keeps the dilated grid (unlike vLLM).
    if dilation != 1 and weight_shape[2:] == (stride,) * 3 and padding == 0 and groups == 1:
        span = dilation * (stride - 1) + 1
        assert res_out.shape[2] == (shape[2] - span) // stride + 1
    for side in (ref_inputs, res_inputs):
        assert_unchanged(side, saved)
    assert_no_alias(res_out, res_inputs)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time)
