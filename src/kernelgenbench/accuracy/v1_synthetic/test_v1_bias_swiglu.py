"""Apply biased SwiGLU with the gate in the first half of the last dimension."""
import kernelgenbench
import pytest
import torch
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import bias_dropout_swiglu_checks as checks


COMBOS = [
    ((1, 2), torch.float32, False, False, False),
    ((2, 7, 30), torch.float32, True, True, False),
    ((2, 9, 34), torch.float16, True, False, False),
    ((3, 11, 66), torch.bfloat16, False, False, False),
    ((2, 128, 1024), torch.float16, False, False, False),
]


def input_build(config):
    return checks.swiglu_inputs(config, "forward")


def compute_ref(y, bias):
    return kernelgenbench.baseline.v1_bias_swiglu(y, bias)


def compute_res(y, bias):
    return kernelgenbench.solution.v1_bias_swiglu(y, bias)


def compare(ref_out, res_out, dtype=None):
    checks.assert_values(ref_out, res_out)


def compute_ref_performance(**inputs):
    return checks.latency_ms(compute_ref, inputs)


def compute_res_performance(**inputs):
    return checks.latency_ms(compute_res, inputs)


@label("v1_bias_swiglu")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_bias_swiglu(config):
    inputs = input_build(config)
    ref_inputs = checks.copy_inputs(inputs, kernelgenbench.device)
    res_inputs = checks.copy_inputs(inputs, kernelgenbench.device)
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out)
    checks.assert_local_contract(ref_inputs, res_inputs, ref_out, res_out)
    fresh_inputs = checks.copy_inputs(inputs, kernelgenbench.device)
    ref_time = compute_ref_performance(**fresh_inputs)
    res_time = compute_res_performance(**fresh_inputs)
    return checks.benchmark_result(ref_time, res_time)
