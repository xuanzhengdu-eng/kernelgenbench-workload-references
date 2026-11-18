"""Accuracy and performance for v1_ds_quantized_linear (DeepSpeed dequant fallback).

Official sources (DeepSpeed pinned commit 8af59b2e7a91c227081617240e98aa282eadc5f9):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/quantization/layers.py#L64-L72
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/quantization/utils.py#L110-L154

Contract (PyTorch fallback branch, selected here with ``group_size=4`` so the
custom CUDA dequantizer is not eligible): asymmetric per-group INT4/INT8
weight dequantization followed by ``torch._C._nn.linear``.  ``x`` is ``[..., H]``
with an optional ``bias`` ``[O]``, both in the input dtype (FP32/FP16/BF16);
packed weights are uint8 ``[O, H]`` (8-bit) or ``[O, H/2]`` (4-bit, high nibble
first); ``group_dim`` 0 groups the output rows and 1 groups the hidden axis,
and ``quant_scale``/``quant_min`` carry one value per group.  Dequantization
happens in the input dtype: codes are cast, divided by the group scale and the
group minimum is added, then the linear runs.  The result is a fresh tensor of
the input dtype and no input is modified.

Item: v1_ds_quantized_linear (723-row 115, fallback contract).  The same entry's
CUDA-dequantizer backend is verified separately in
``test_v1_ds_quantized_linear_cuda.py``; the INT8 ``vector_matmul`` mode is a
different contract and lives in ``test_v1_ds_vector_matmul_qint8.py``.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows112_121_checks as checks


# (input prefix, O, H, bits, group_dim, bias, dtype, padded-input layout):
# x is (*prefix, H); the padded variant takes every second lane of a 2H buffer.
# group_size is fixed at 4, which forces the Python dequantizer.
CASES = [
    ((3,), 8, 8, 4, 1, True, torch.float32, False),
    ((2, 3), 8, 12, 8, 1, False, torch.float32, True),
    ((2,), 8, 8, 4, 0, True, torch.float16, True),
    ((2, 2), 12, 8, 8, 0, False, torch.bfloat16, False),
    ((1,), 4, 16, 4, 1, False, torch.bfloat16, True),
    ((3,), 4, 8, 8, 1, True, torch.float16, False),
    ((1, 2), 8, 16, 4, 0, False, torch.float32, False),
    ((2,), 12, 12, 8, 0, True, torch.float32, True),
]

# Axes: one combo per case (dtype is part of the case).
COMBOS = list(CASES)

GROUP_SIZE = 4

# Dequantization rounds through the input dtype once; the linear then accumulates
# in that dtype, so the allowance follows the storage type.
TOLERANCE = {
    torch.float32: (3e-4, 3e-5),
    torch.float16: (4e-2, 6e-3),
    torch.bfloat16: (1e-1, 1e-2),
}


def input_build(config):
    """x [*prefix, H]; q [O, H] uint8 (8-bit) or [O, H/2] (4-bit);
    scale/min [O/G, 1, H] for group_dim=0 or [O, H/G, 1] for group_dim=1.
    """
    prefix, out_features, hidden, bits, group_dim, has_bias, dtype, padded = config
    count = 1
    for size in prefix:
        count *= size
    width = hidden * (2 if padded else 1)
    x = (((torch.arange(count * width, dtype=torch.float32) * 7) % 43 - 21)
         / 17).reshape(*prefix, width).to(dtype)
    if padded:
        x = x[..., ::2]
    values = ((torch.arange(out_features * hidden, dtype=torch.int32) * 7 + 3)
              % (16 if bits == 4 else 256)).to(torch.uint8).reshape(out_features, hidden)
    packed = (((values[:, ::2] << 4) | values[:, 1::2]) if bits == 4 else values)
    shape = (out_features // GROUP_SIZE, 1, hidden) if group_dim == 0 \
        else (out_features, hidden // GROUP_SIZE, 1)
    total = 1
    for size in shape:
        total *= size
    scale = (torch.arange(total, dtype=torch.float32).reshape(shape) % 5 / 4 + 1.25).to(dtype)
    minimum = (torch.arange(total, dtype=torch.float32).reshape(shape) % 7 / 8 - 0.5).to(dtype)
    bias = (torch.arange(out_features, dtype=torch.float32) / 11 - 0.2).to(dtype) \
        if has_bias else None
    return dict(x=x, quantized_weight=packed, quant_scale=scale, quant_min=minimum,
                bias=bias, num_bits=bits, group_size=GROUP_SIZE, group_dim=group_dim)


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_ds_quantized_linear(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_ds_quantized_linear(**inputs)


def compare(ref_out, res_out, dtype):
    assert isinstance(res_out, torch.Tensor), f"expected a tensor, got {type(res_out).__name__}"
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == dtype
    assert res_out.device == ref_out.device
    rtol, atol = TOLERANCE[dtype]
    torch.testing.assert_close(res_out.float(), ref_out.float(), rtol=rtol, atol=atol)


def compute_ref_performance(**inputs):
    return checks.measure(kernelgenbench.baseline.v1_ds_quantized_linear,
                          inputs, kernelgenbench.device)


def compute_res_performance(**inputs):
    return checks.measure(kernelgenbench.solution.v1_ds_quantized_linear,
                          inputs, kernelgenbench.device)


@label("v1_ds_quantized_linear")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_ds_quantized_linear(config):
    prefix, out_features, hidden, bits, group_dim, has_bias, dtype, padded = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    assert res_inputs["x"].shape == (*prefix, hidden)
    assert res_inputs["x"].is_contiguous() == (not padded)
    assert res_inputs["quantized_weight"].shape == (
        out_features, hidden // 2 if bits == 4 else hidden)
    assert res_inputs["quant_scale"].shape == (
        (out_features // GROUP_SIZE, 1, hidden) if group_dim == 0
        else (out_features, hidden // GROUP_SIZE, 1))

    # Dequantized weights are asymmetric here: the group minimum must shift them.
    assert float(inputs["quant_min"].abs().max()) > 0

    before = {name: value.clone() for name, value in res_inputs.items()
              if isinstance(value, torch.Tensor)}
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    for name, value in before.items():
        assert torch.equal(value, res_inputs[name]), f"input {name} was modified"
        assert not torch._C._is_alias_of(res_out, res_inputs[name]), f"output aliases {name}"
    return checks.timings(compute_ref_performance(**inputs),
                          compute_res_performance(**inputs))
