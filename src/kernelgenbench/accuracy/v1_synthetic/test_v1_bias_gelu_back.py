"""Accuracy and performance for v1_bias_gelu_back."""
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
    y = checks.input_tensor(shape, dtype, generator, strided=layout == "strided")
    bias = checks.input_tensor((shape[-1],), dtype, generator)
    g = checks.input_tensor(shape, dtype, generator)
    return {
        "g": g,
        "bias": bias,
        "y": y,
    }


def compute_ref(g, bias, y):
    return kernelgenbench.baseline.v1_bias_gelu_back(g, bias, y)


def compute_res(g, bias, y):
    return kernelgenbench.solution.v1_bias_gelu_back(g, bias, y)


def compare(ref_out, res_out, dtype=None):
    checks.assert_values(res_out, ref_out, dtype)


def compute_ref_performance(g, bias, y):
    return checks.measure(kernelgenbench.baseline.v1_bias_gelu_back, {"g": g, "bias": bias, "y": y})


def compute_res_performance(g, bias, y):
    return checks.measure(kernelgenbench.solution.v1_bias_gelu_back, {"g": g, "bias": bias, "y": y})


@label("v1_bias_gelu_back")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_bias_gelu_back(config):
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
