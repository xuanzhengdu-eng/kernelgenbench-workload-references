"""Accuracy and performance for v1_ds_quantized_linear_cuda (DeepSpeed CUDA dequantizer).

Official sources (DeepSpeed pinned commit 8af59b2e7a91c227081617240e98aa282eadc5f9):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/quantization/utils.py#L110-L146
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/quantization/quantize_intX.cu#L195-L235

Second backend of the same ``QuantizedLinear.forward`` entry: for FP16 input
with ``group_dim`` on the last axis and ``group_size % 8 == 0`` the dequantizer
calls ``dequantize_int4_to_half_experimental`` / ``dequantize_int8_to_half_experimental``.
One thread converts four packed bytes (int4) or eight bytes (int8), divides by
the group scale and adds the group minimum, all in FP16, then
``torch._C._nn.linear`` runs.  The kernel launches ``num_group / 32`` blocks, so
only sizes with at least 32 groups (``O * H / group_size``) are usable; that
grid limit is an upstream precondition, not a contract change.  The output is a
fresh FP16 tensor and no input is modified.

Item: v1_ds_quantized_linear (723-row 115, CUDA-dequantizer backend).  Verified
with the real pinned DeepSpeed extension; the verification log records that
``get_quantizer_module`` was reached and which dequantize kernel launched.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows112_121_checks as checks


# (input prefix, O, H, bits, group_size, bias): FP16 only, group_dim is the last
# axis, group_size is a multiple of 8, and O*H/group_size >= 32.
CASES = [
    ((16,), 32, 64, 4, 8, True),
    ((4, 8), 32, 64, 8, 8, False),
    ((8,), 64, 128, 4, 16, True),
    ((2, 3, 4), 32, 64, 4, 8, False),
    ((32,), 16, 128, 8, 8, True),
]

# Axes: one combo per case.
COMBOS = list(CASES)

TOLERANCE = (1e-3, 1e-3)


def input_build(config):
    """x [*prefix, H] fp16; q [O, H/2] or [O, H] uint8; scale/min [O, H/G, 1] fp16."""
    prefix, out_features, hidden, bits, group_size, has_bias = config
    count = 1
    for size in prefix:
        count *= size
    x = (((torch.arange(count * hidden, dtype=torch.float32) % 37 - 18) / 9)
         .reshape(*prefix, hidden).to(torch.float16))
    values = ((torch.arange(out_features * hidden, dtype=torch.int32) * 7 + 3)
              % (16 if bits == 4 else 256)).to(torch.uint8).reshape(out_features, hidden)
    packed = (((values[:, ::2] << 4) | values[:, 1::2]) if bits == 4 else values)
    shape = (out_features, hidden // group_size, 1)
    total = out_features * (hidden // group_size)
    assert total >= 32, "the pinned kernel launches num_group/32 blocks"
    scale = (torch.arange(total, dtype=torch.float32).reshape(shape) % 5 / 4 + 1.25).half()
    minimum = (torch.arange(total, dtype=torch.float32).reshape(shape) % 7 / 8 - 0.5).half()
    bias = (torch.arange(out_features, dtype=torch.float32) / 11 - 0.2).half() \
        if has_bias else None
    return dict(x=x, quantized_weight=packed, quant_scale=scale, quant_min=minimum,
                bias=bias, num_bits=bits, group_size=group_size)


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_ds_quantized_linear_cuda(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_ds_quantized_linear_cuda(**inputs)


def compare(ref_out, res_out):
    assert isinstance(res_out, torch.Tensor), f"expected a tensor, got {type(res_out).__name__}"
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == torch.float16
    assert res_out.device == ref_out.device
    rtol, atol = TOLERANCE
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)


def compute_ref_performance(**inputs):
    return checks.measure(kernelgenbench.baseline.v1_ds_quantized_linear_cuda,
                          inputs, kernelgenbench.device)


def compute_res_performance(**inputs):
    return checks.measure(kernelgenbench.solution.v1_ds_quantized_linear_cuda,
                          inputs, kernelgenbench.device)


@label("v1_ds_quantized_linear_cuda")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_ds_quantized_linear_cuda(config):
    prefix, out_features, hidden, bits, group_size, has_bias = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    assert res_inputs["x"].shape == (*prefix, hidden)
    assert res_inputs["quantized_weight"].shape == (
        out_features, hidden // 2 if bits == 4 else hidden)
    assert res_inputs["quant_scale"].shape == (out_features, hidden // group_size, 1)

    before = {name: value.clone() for name, value in res_inputs.items()
              if isinstance(value, torch.Tensor)}
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out)

    for name, value in before.items():
        assert torch.equal(value, res_inputs[name]), f"input {name} was modified"
        assert not torch._C._is_alias_of(res_out, res_inputs[name]), f"output aliases {name}"

    return checks.timings(compute_ref_performance(**inputs),
                          compute_res_performance(**inputs))
