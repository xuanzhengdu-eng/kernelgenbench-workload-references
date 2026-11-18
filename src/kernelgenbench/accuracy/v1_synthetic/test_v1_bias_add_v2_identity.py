"""DeepSpeed CUDABiasActivation configured as IDENTITY, mutates activation and returns None."""
import kernelgenbench
import pytest
import torch

from kernelgenbench.accuracy.v1_synthetic._bias_residual_checks import (
    benchmark_call, check_state, copy_inputs, result_from_times,
)
from sandbox.verifier.test_parametrize import label, parametrize


COMBOS = [
    (shape, dtype, has_bias)
    for shape in ((1, 8), (32, 128), (256, 1024))
    for dtype in (torch.float16, torch.bfloat16)
    for has_bias in (True, False)
]


def input_build(config):
    shape, dtype, has_bias = config
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    activation = torch.randn(shape, generator=generator, dtype=dtype, device="cpu")
    activation[0, 0] = -0.0
    bias = torch.randn(shape[1], generator=generator, dtype=dtype, device="cpu") if has_bias else None
    return {"activation": activation, "bias": bias}


def compute_ref(activation, bias):
    return kernelgenbench.baseline.v1_bias_add_v2_identity(activation, bias)


def compute_res(activation, bias):
    return kernelgenbench.solution.v1_bias_add_v2_identity(activation, bias)


def compare(ref_out, res_out, dtype=None):
    # A raw None from an output-only remote evaluator contains no evidence of the write.
    if not all(isinstance(result, tuple) and len(result) == 2 for result in (ref_out, res_out)):
        raise AssertionError("IDENTITY requires an observation of (return value, post-call activation)")
    ref_value, ref_activation = ref_out
    res_value, res_activation = res_out
    assert ref_value is None and res_value is None, "both entries must return None"
    torch.testing.assert_close(res_activation, ref_activation, rtol=0, atol=0, check_stride=True)
    assert torch.equal(torch.signbit(res_activation), torch.signbit(ref_activation))


def compute_ref_performance(**inputs):
    return benchmark_call(compute_ref, inputs)


def compute_res_performance(**inputs):
    return benchmark_call(compute_res, inputs)


@label("v1_bias_add_v2_identity")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_v1_bias_add_v2_identity(config):
    inputs = copy_inputs(input_build(config), kernelgenbench.device)
    expected = copy_inputs(inputs, kernelgenbench.device)
    actual = copy_inputs(inputs, kernelgenbench.device)
    ref_out = compute_ref(**expected)
    res_out = compute_res(**actual)
    compare((ref_out, expected["activation"]), (res_out, actual["activation"]))
    check_state(inputs, expected, actual, ref_out, res_out)
    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return result_from_times(ref_time, res_time)
