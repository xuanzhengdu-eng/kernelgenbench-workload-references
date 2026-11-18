"""Accuracy and performance for vLLM's multimodal per-channel affine transform."""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic import rows42_51_test_utils as u
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

# Two input layouts: "pcl" = (patches, C*L) flattened pixels, "bchw" =
# (batch, C, H, W) images.  weight/bias are fp32 per-channel constants, the
# arithmetic is fp32, and the output is visual_dtype with the input shape.
PCL = (4, 3 * 8)
BCHW = (2, 3, 4, 4)
COMBOS = [
    ("pcl", (4, 3 * 8), torch.uint8, torch.bfloat16),
    ("pcl", (4, 3 * 8), torch.uint8, torch.float16),
    ("pcl", (4, 3 * 8), torch.float32, torch.bfloat16),
    ("pcl", (4, 3 * 8), torch.float16, torch.bfloat16),
    ("bchw", (2, 3, 4, 4), torch.uint8, torch.bfloat16),
    ("bchw", (2, 3, 4, 4), torch.float32, torch.bfloat16),
    ("bchw", (2, 3, 4, 4), torch.uint8, torch.float32),
]


def input_build(config):
    layout, shape, input_dtype, visual_dtype = config
    generator = torch.Generator(device="cpu").manual_seed(368)
    channels = shape[1] if layout == "bchw" else 3
    if input_dtype == torch.uint8:
        pixel_values = torch.randint(0, 256, shape, generator=generator, dtype=torch.uint8)
    else:
        pixel_values = torch.randn(shape, dtype=input_dtype, generator=generator) * 0.5
    weight = torch.rand(channels, generator=generator, dtype=torch.float32) + 0.5
    bias = torch.randn(channels, generator=generator, dtype=torch.float32) * 0.1
    return {"pixel_values": pixel_values, "weight": weight, "bias": bias,
            "visual_dtype": visual_dtype}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_mm_channel_affine(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_mm_channel_affine(**inputs)


def compare(ref_out, res_out, shape, visual_dtype):
    assert isinstance(res_out, torch.Tensor) and isinstance(ref_out, torch.Tensor)
    assert res_out.shape == ref_out.shape == shape
    assert res_out.dtype == ref_out.dtype == visual_dtype
    rtol, atol = u.tolerance(visual_dtype)
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)


def compute_ref_performance(**inputs):
    return u.measure(compute_ref, u.fresh(inputs, kernelgenbench.device))


def compute_res_performance(**inputs):
    return u.measure(compute_res, u.fresh(inputs, kernelgenbench.device))


@label("v1_mm_channel_affine")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_mm_channel_affine(config):
    layout, shape, input_dtype, visual_dtype = config
    inputs = input_build(config)
    ref_inputs = u.fresh(inputs, kernelgenbench.device)
    res_inputs = u.fresh(inputs, kernelgenbench.device)
    before = u.clone_tensors(res_inputs)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, shape, visual_dtype)

    # Channel 0 is scaled and shifted independently of the other channels.
    channels = res_inputs["weight"].numel()
    x3 = res_inputs["pixel_values"].reshape(shape[0], channels, -1).float()
    expected0 = x3[:, 0] * res_inputs["weight"][0] + res_inputs["bias"][0]
    actual0 = res_out.reshape(shape[0], channels, -1)[:, 0].float()
    rtol, atol = u.tolerance(visual_dtype)
    torch.testing.assert_close(actual0, expected0, rtol=rtol, atol=atol)

    u.assert_unchanged(before, res_inputs)
    u.assert_fresh(res_out, res_inputs)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
