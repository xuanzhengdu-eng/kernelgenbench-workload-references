"""No-bias forward via VectorAddOp PyTorch fallback; result has the dtype of residual + attention_output."""
import kernelgenbench
import pytest
import torch

from kernelgenbench.accuracy.v1_synthetic._bias_residual_checks import (
    benchmark_call, check_state, copy_inputs, result_from_times,
)
from sandbox.verifier.test_parametrize import label, parametrize


COMBOS = [
    ((2, 3, 32), torch.float32, torch.float32, 2), ((1, 2, 64), torch.float32, torch.float32, 4),
    ((1, 1, 7), torch.float16, torch.float32, 3), ((2, 17, 121), torch.bfloat16, torch.bfloat16, 4),
    ((2, 128, 1024), torch.float16, torch.float16, 2),
]


def input_build(config):
    shape, residual_dtype, hidden_dtype, mp = config
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    def sample(dtype):
        return torch.randn(shape, generator=generator, device="cpu", dtype=dtype)
    return {"hidden_state": sample(hidden_dtype), "residual": sample(residual_dtype),
            "attention_output": sample(residual_dtype), "mp_size": mp}


def compute_ref(hidden_state, residual, attention_output, mp_size):
    return kernelgenbench.baseline.v1_ds_attention_residual_no_bias(hidden_state, residual, attention_output, mp_size)


def compute_res(hidden_state, residual, attention_output, mp_size):
    return kernelgenbench.solution.v1_ds_attention_residual_no_bias(hidden_state, residual, attention_output, mp_size)


def compare(ref_out, res_out, dtype=None):
    torch.testing.assert_close(res_out, ref_out, check_stride=True, atol=0, rtol=0)


def compute_ref_performance(**inputs):
    return benchmark_call(compute_ref, inputs)


def compute_res_performance(**inputs):
    return benchmark_call(compute_res, inputs)


@label("v1_ds_attention_residual_no_bias")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_v1_ds_attention_residual_no_bias(config):
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
