"""Accuracy and performance for v1_ds_bias_silu_inplace."""
import kernelgenbench
import pytest
import torch
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import bias_activation_test_utils as checks


COMBOS = [
    (shape, dtype, layout)
    for shape in [(1, 8), (7, 24), (32, 128), (256, 1024)]
    for dtype in (torch.float16, torch.bfloat16)
    for layout in ("bias", "no_bias")
]


def input_build(config):
    shape, dtype, layout = config
    generator = torch.Generator(device="cpu").manual_seed(73)
    activation = checks.input_tensor(shape, dtype, generator)
    bias = None if layout == "no_bias" else checks.input_tensor((shape[-1],), dtype, generator)
    # Large negative values exercise the float exponential before rounding.
    activation[0, :6] = torch.tensor([-100.0, -0.0, 1.0, -2.0, 40.0, float("nan")], dtype=dtype)
    if bias is not None:
        bias[:6] = -0.0
    return {
        "activation": activation,
        "bias": bias,
    }


def compute_ref(activation, bias):
    return checks.observe_update(kernelgenbench.baseline.v1_ds_bias_silu_inplace, activation, bias)


def compute_res(activation, bias):
    return checks.observe_update(kernelgenbench.solution.v1_ds_bias_silu_inplace, activation, bias)


def compare(ref_out, res_out, dtype=None):
    checks.compare_updates(ref_out, res_out, dtype, returns_none=True)
    checks.check_zero_sign(res_out["activation_after"], ref_out["activation_after"])


def compute_ref_performance(activation, bias):
    return checks.measure(kernelgenbench.baseline.v1_ds_bias_silu_inplace, {"activation": activation, "bias": bias})


def compute_res_performance(activation, bias):
    return checks.measure(kernelgenbench.solution.v1_ds_bias_silu_inplace, {"activation": activation, "bias": bias})


@label("v1_ds_bias_silu_inplace")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_ds_bias_silu_inplace(config):
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype=config[1])

    return checks.timings(compute_ref_performance(**inputs), compute_res_performance(**inputs))
