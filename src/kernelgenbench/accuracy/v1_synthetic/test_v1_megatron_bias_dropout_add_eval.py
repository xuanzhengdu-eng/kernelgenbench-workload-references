"""Megatron inference dropout, aliases x only with matching dtype and all inputs disabling gradients. Rank-3 input and FP32 residual, with optional FP16/BF16 x and bias promoted to FP32."""
import kernelgenbench
import pytest
import torch
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import bias_dropout_swiglu_checks as checks


# The original synthetic probability domain is [0, 1).
COMBOS = [
    ((1, 1, 1), 0.0, False, "none", torch.float32),
    ((2, 5, 17), 0.0, True, "x", torch.float32),
    ((2, 7, 31), 0.5, True, "bias", torch.float32),
    ((3, 9, 15), 0.5, False, "residual", torch.float32),
    ((2, 63, 127), 0.25, True, "none", torch.float32),
    ((2, 64, 129), 0.5, False, "none", torch.float32),
    ((4, 128, 256), 0.75, True, "none", torch.float32),
    ((4, 127, 259), 0.95, True, "none", torch.float32),
    ((2, 61, 133), 0.2, True, "none", torch.float16),
    ((2, 59, 135), 0.5, False, "none", torch.bfloat16),
    ((3, 11, 33), 0.0, True, "x", torch.bfloat16),
]


def input_build(config):
    return checks.dropout_inputs(config, megatron=True)


def compute_ref(x_with_bias, residual, prob):
    return kernelgenbench.baseline.v1_megatron_bias_dropout_add_eval(x_with_bias, residual, prob)


def compute_res(x_with_bias, residual, prob):
    return kernelgenbench.solution.v1_megatron_bias_dropout_add_eval(x_with_bias, residual, prob)


def compare(ref_out, res_out, dtype=None):
    checks.assert_values(ref_out, res_out)


def compute_ref_performance(**inputs):
    return checks.latency_ms(compute_ref, inputs)


def compute_res_performance(**inputs):
    return checks.latency_ms(compute_res, inputs)


@label("v1_megatron_bias_dropout_add_eval")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_bias_dropout_add_eval(config):
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
