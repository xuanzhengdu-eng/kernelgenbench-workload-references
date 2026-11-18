"""Accuracy and performance for v1_quant_compat_concat.

Operator: DeepSpeed ``concat_to_compat_param`` (quantized-weight compatibility
packaging with attached inverse views).
Official source (pinned commit 8af59b2e7a91c227081617240e98aa282eadc5f9):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/quantization/utils.py#L191-L221

Contract: flattens ``quantized_weight`` [R, C], ``quant_scale`` [R, 1] and
``quant_min`` [R, 1] and concatenates them in that order.  ``return_param``
wraps the 1-D result in a non-trainable ``nn.Parameter``; otherwise a plain
Tensor is returned.  The result carries a ``deconcat(tensor)`` callable that
returns the three original shapes as views into the package storage (so writing
a view changes the package).  Inputs are not modified.

Item: v1_quant_compat_concat (723-row 634).
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic.rows632_641_test_utils import (
    fresh, measure, timings)

# (rows, cols) for the quantized weight; scale/min are (rows, 1).
SHAPES = [(4, 8), (3, 16)]
COMBOS = [(shape, dtype, return_param) for shape in SHAPES
          for dtype in (torch.float32, torch.bfloat16)
          for return_param in (False, True)]


def input_build(config):
    """weight: [R, C], scale: [R, 1], min: [R, 1]; all same dtype."""
    (rows, cols), dtype, return_param = config
    generator = torch.Generator(device="cpu").manual_seed(20261109)

    def draw(shape):
        return torch.randn(shape, dtype=dtype, generator=generator)

    return {
        "quantized_weight": draw((rows, cols)),
        "quant_scale": draw((rows, 1)),
        "quant_min": draw((rows, 1)),
        "return_param": return_param,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_quant_compat_concat(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_quant_compat_concat(**inputs)


def compare(ref_out, res_out, dtype, return_param, shape):
    assert isinstance(ref_out, torch.Tensor)
    assert isinstance(res_out, torch.Tensor), "candidate must return a tensor"
    assert isinstance(res_out, torch.nn.Parameter) == return_param, \
        "return_param must control the Parameter wrapper"
    if return_param:
        assert not res_out.requires_grad
    rows, cols = shape
    assert res_out.shape == ref_out.shape == (rows * cols + 2 * rows,)
    assert res_out.dtype == ref_out.dtype == dtype
    torch.testing.assert_close(res_out.float(), ref_out.float(), rtol=0, atol=0)

    # The attached inverse views must reconstruct the three inputs and alias the package.
    assert hasattr(res_out, "deconcat") and hasattr(ref_out, "deconcat")
    ref_parts = ref_out.deconcat(ref_out)
    res_parts = res_out.deconcat(res_out)
    assert len(res_parts) == len(ref_parts) == 3
    for index, (left, right) in enumerate(zip(ref_parts, res_parts)):
        assert right.shape == left.shape, f"part {index}: shape"
        assert right.dtype == left.dtype, f"part {index}: dtype"
        torch.testing.assert_close(right.float(), left.float(), rtol=0, atol=0)
        assert right.untyped_storage().data_ptr() == res_out.untyped_storage().data_ptr(), \
            f"part {index} must be a view of the returned package"


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs, kernelgenbench.device)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs, kernelgenbench.device)


@label("v1_quant_compat_concat")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_quant_compat_concat(config):
    shape, dtype, return_param = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    ref_before = {name: value.clone() for name, value in ref_inputs.items()
                  if isinstance(value, torch.Tensor)}
    res_before = {name: value.clone() for name, value in res_inputs.items()
                  if isinstance(value, torch.Tensor)}

    compare(compute_ref(**ref_inputs), compute_res(**res_inputs), dtype,
            return_param, shape)

    # Packaging must not consume or modify the three source tensors.
    for name, before in ref_before.items():
        torch.testing.assert_close(ref_inputs[name].cpu(), before.cpu(), rtol=0, atol=0)
    for name, before in res_before.items():
        torch.testing.assert_close(res_inputs[name].cpu(), before.cpu(), rtol=0, atol=0)
    return timings(compute_ref_performance(**inputs),
                   compute_res_performance(**inputs))
