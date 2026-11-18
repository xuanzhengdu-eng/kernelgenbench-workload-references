"""Accuracy and performance for v1_ds_fpdt_bias_gelu_back."""
import kernelgenbench
import pytest
import torch
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import bias_activation_test_utils as checks


COMBOS = [
    (shape, dtype, layout)
    for shape in [(0, 13), (3, 7), (32, 128), (256, 1024), (2, 16, 128), (2, 128, 1024)]
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
    for layout in ("contiguous", "strided")
    if layout != "strided" or min(shape[:2]) > 1
]


def input_build(config):
    shape, dtype, layout = config
    generator = torch.Generator(device="cpu").manual_seed(73)
    x = checks.input_tensor(shape, dtype, generator, strided=layout == "strided")
    g = checks.input_tensor(shape, dtype, generator)
    return {
        "g": g,
        "x": x,
    }


def compute_ref(g, x):
    return kernelgenbench.baseline.v1_ds_fpdt_bias_gelu_back(g, x)


def compute_res(g, x):
    return kernelgenbench.solution.v1_ds_fpdt_bias_gelu_back(g, x)


def compare(ref_out, res_out, dtype=None):
    checks.assert_values(res_out, ref_out, dtype)


def compute_ref_performance(g, x):
    return checks.measure(kernelgenbench.baseline.v1_ds_fpdt_bias_gelu_back, {"g": g, "x": x})


def compute_res_performance(g, x):
    return checks.measure(kernelgenbench.solution.v1_ds_fpdt_bias_gelu_back, {"g": g, "x": x})


@label("v1_ds_fpdt_bias_gelu_back")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_ds_fpdt_bias_gelu_back(config):
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype=config[1])
    original = {name: value.to(kernelgenbench.device) for name, value in inputs.items()}
    checks.validate_unchanged(original, res_inputs)
    checks.check_fresh_result(res_out, res_inputs)

    return checks.timings(compute_ref_performance(**inputs), compute_res_performance(**inputs))
