"""Accuracy and performance for v1_vllm_pack_image_patches (CSV row 449, Pixtral).

Contract (vLLM ``pixtral.pack_image_patches``):
  images: list of 3-D (channels, H, W) tensors, all with the same channel count
  and dtype; patch_shape is (ph, pw) and H/W are truncated to whole patches.
  Returns (packed, grids) where packed is a fresh contiguous
  (sum(rows * cols), channels * ph * pw) tensor, row-major per image, each row
  flattened as (channels, ph, pw), and grids lists each image's (rows, cols).
  The images are not modified.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows442_451_test_utils import (
    assert_unchanged,
    measure,
    move_inputs,
    tolerance,
)
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (image (H, W) list, channels, patch (ph, pw), dtype, strided width)
COMBOS = [
    (((4, 6), (6, 4)), 3, (2, 2), torch.float32, False),
    (((5, 7), (4, 8)), 2, (2, 3), torch.bfloat16, False),
    (((4, 4),), 1, (2, 2), torch.float32, True),
    (((6, 8), (3, 5)), 3, (3, 2), torch.bfloat16, True),
    (((2, 2),), 4, (7, 7), torch.float16, False),   # patch larger than the image
]


def input_build(config):
    """CHW images built from a deterministic ramp; strided keeps every other column."""
    shapes, channels, patch, dtype, strided = config
    images = []
    for index, (height, width) in enumerate(shapes):
        source_width = width * 2 if strided else width
        values = (torch.arange(channels * height * source_width, dtype=torch.float32)
                  + index * 3)
        image = values.reshape(channels, height, source_width).to(dtype)
        images.append(image[:, :, ::2] if strided else image)
    return {"images": images, "patch_shape": patch}


def compute_ref(images, patch_shape):
    return kernelgenbench.baseline.v1_vllm_pack_image_patches(images, patch_shape)


def compute_res(images, patch_shape):
    return kernelgenbench.solution.v1_vllm_pack_image_patches(images, patch_shape)


def _expected(images, patch_shape):
    """Independent expectation: slice each patch and flatten (C, ph, pw)."""
    patch_height, patch_width = patch_shape
    rows = []
    grids = []
    for image in images:
        channels, height, width = image.shape
        grid = (height // patch_height, width // patch_width)
        grids.append(grid)
        for row in range(grid[0]):
            for column in range(grid[1]):
                patch = image[:, row * patch_height:(row + 1) * patch_height,
                              column * patch_width:(column + 1) * patch_width]
                rows.append(patch.reshape(-1))
    if not rows:
        return torch.empty((0, images[0].shape[0] * patch_height * patch_width),
                           dtype=images[0].dtype, device=images[0].device), grids
    return torch.stack(rows), grids


def compare(ref_out, res_out, config, ref_inputs, res_inputs):
    patch_shape, dtype = config[2], config[3]
    expected, grids = _expected(ref_inputs["images"], patch_shape)
    width = ref_inputs["images"][0].shape[0] * patch_shape[0] * patch_shape[1]
    for output, inputs in ((ref_out, ref_inputs), (res_out, res_inputs)):
        assert isinstance(output, tuple) and len(output) == 2
        packed, grid = output
        assert isinstance(packed, torch.Tensor) and packed.dtype == dtype
        assert packed.shape == (expected.shape[0], width)
        assert grid == grids
        assert packed.is_contiguous()
        torch.testing.assert_close(packed, expected, rtol=0, atol=0)
        for image in inputs["images"]:
            assert not torch._C._is_alias_of(packed, image)
    assert torch.equal(ref_out[0], res_out[0])


def compute_ref_performance(images, patch_shape):
    return measure(compute_ref, {"images": images, "patch_shape": patch_shape})


def compute_res_performance(images, patch_shape):
    return measure(compute_res, {"images": images, "patch_shape": patch_shape})


@label("v1_vllm_pack_image_patches")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_pack_image_patches(config):
    shapes, channels, patch_shape, dtype, strided = config
    inputs = move_inputs(input_build(config), kernelgenbench.device)
    ref_inputs = move_inputs(input_build(config), kernelgenbench.device)
    res_inputs = move_inputs(input_build(config), kernelgenbench.device)
    before = [image.clone() for image in inputs["images"]]
    for side in (ref_inputs, res_inputs):
        for image, (height, width) in zip(side["images"], shapes):
            assert image.shape == (channels, height, width)
            assert image.dtype == dtype
            assert image.is_contiguous() == (not strided)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config, ref_inputs, res_inputs)
    assert_unchanged(inputs, {"images": before})

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
