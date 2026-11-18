"""Post-norm ResidualAddOp.forward with a supplied replacement buffer and PyTorch fallback."""
import kernelgenbench
import pytest
import torch

from kernelgenbench.accuracy.v1_synthetic._bias_residual_checks import (
    benchmark_call, check_state, copy_inputs, result_from_times,
)
from sandbox.verifier.test_parametrize import label, parametrize


COMBOS = [
    ((2, 2, 32), torch.float32, 2, True, True),
    ((1, 4, 64), torch.float32, 1, True, False),
    ((2, 2, 64), torch.float32, 2, False, False),
    ((1, 3, 32), torch.float32, 4, False, True),
    ((1, 1, 8), torch.float16, 1, False, True),
    ((2, 17, 120), torch.bfloat16, 2, True, False),
    ((2, 128, 1024), torch.float16, 4, True, True),
]


def input_build(config):
    shape, dtype, mp, mlp, add = config
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    def sample(size):
        return torch.randn(size, generator=generator, device="cpu", dtype=dtype)
    return {
        "hidden_state": sample(shape), "residual": sample(shape), "attention_output": sample(shape),
        "attention_bias": sample((shape[-1],)), "final_bias": sample((shape[-1],)), "mp_size": mp,
        "mlp_after_attn": mlp, "add_attn_bias": add, "residual_add": sample(shape),
    }


def compute_ref(hidden_state, residual, attention_output, attention_bias, final_bias, residual_add, mp_size, mlp_after_attn, add_attn_bias):
    return kernelgenbench.baseline.v1_ds_attention_residual_substituted(hidden_state, residual, attention_output, attention_bias, final_bias, residual_add, mp_size, mlp_after_attn, add_attn_bias)


def compute_res(hidden_state, residual, attention_output, attention_bias, final_bias, residual_add, mp_size, mlp_after_attn, add_attn_bias):
    return kernelgenbench.solution.v1_ds_attention_residual_substituted(hidden_state, residual, attention_output, attention_bias, final_bias, residual_add, mp_size, mlp_after_attn, add_attn_bias)


def compare(ref_out, res_out, dtype=None):
    torch.testing.assert_close(res_out, ref_out, check_stride=True, atol=0, rtol=0)


def compute_ref_performance(**inputs):
    return benchmark_call(compute_ref, inputs)


def compute_res_performance(**inputs):
    return benchmark_call(compute_res, inputs)


@label("v1_ds_attention_residual_substituted")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_v1_ds_attention_residual_substituted(config):
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
