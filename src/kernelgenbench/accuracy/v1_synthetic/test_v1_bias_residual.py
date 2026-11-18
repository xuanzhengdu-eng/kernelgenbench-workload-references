"""Spatial three-input FP16 kernel with independent fresh output."""
import kernelgenbench
import pytest
import torch

from kernelgenbench.accuracy.v1_synthetic._bias_residual_checks import (
    benchmark_call, check_state, copy_inputs, result_from_times,
)
from sandbox.verifier.test_parametrize import label, parametrize


COMBOS = [(1, 1, 8), (2, 7, 40), (1, 16, 64), (2, 128, 768), (2, 512, 1024), (2, 16, 5, 7)]


def input_build(config):
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    def sample(shape):
        return torch.randn(shape, generator=generator, dtype=torch.float16, device="cpu")
    activation, other = sample(config), sample(config)
    channels = config[-1]
    if len(config) == 4:
        activation = activation.contiguous(memory_format=torch.channels_last)
        other = other.contiguous(memory_format=torch.channels_last)
        channels = config[1]
    return {"activation": activation, "bias": sample((channels,)), "other": other}


def compute_ref(activation, bias, other):
    return kernelgenbench.baseline.v1_bias_residual(activation, bias, other)


def compute_res(activation, bias, other):
    return kernelgenbench.solution.v1_bias_residual(activation, bias, other)


def compare(ref_out, res_out, dtype=None):
    torch.testing.assert_close(res_out, ref_out, check_stride=True, atol=0, rtol=0)


def compute_ref_performance(**inputs):
    return benchmark_call(compute_ref, inputs)


def compute_res_performance(**inputs):
    return benchmark_call(compute_res, inputs)


@label("v1_bias_residual")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_v1_bias_residual(config):
    inputs = copy_inputs(input_build(config), kernelgenbench.device)
    expected = copy_inputs(inputs, kernelgenbench.device)
    actual = copy_inputs(inputs, kernelgenbench.device)
    ref_out = compute_ref(**expected)
    res_out = compute_res(**actual)
    compare(ref_out, res_out)
    check_state(inputs, expected, actual, ref_out, res_out)
    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return result_from_times(ref_time, res_time)
