"""Accuracy and performance for v1_ds_grouped_asymmetric_quantize.

Operator: DeepSpeed ``Quantizer(config).quantize`` with ``symmetric=False``.

Official source (pinned commit):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/quantization/utils.py#L48-L99

Contract: group-wise asymmetric quantization on ``group_dim``.  Returns
``(quantized_uint8, scale, group_min)``; 4-bit output packs two values per byte
(high nibble first), 8-bit output keeps one byte per value.  The input tensor is
cloned and never modified.  The grid crosses three shapes/group axes, both bit
widths and all three float dtypes the pinned Python implementation accepts.

``compute_ref`` loads the transcribed reference from ``kernelgenbench.baseline``;
``compute_res`` calls the operator bound as ``kernelgenbench.solution``, i.e. the
real pinned ``Quantizer`` during verification.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows702_711_checks as checks


# (shape, group_dim, group_size); the grouped axis is divisible by group_size.
CASES = [
    ((4, 8), 0, 2),
    ((4, 8), 1, 4),
    ((8, 16), 1, 8),
]
COMBOS = [
    (shape, dim, size, dtype, num_bits)
    for shape, dim, size in CASES
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
    for num_bits in (4, 8)
]


def input_build(config):
    """One CPU tensor with the configured shape and dtype."""
    shape, group_dim, group_size, dtype, num_bits = config
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    return {
        "tensor": torch.randn(shape, dtype=dtype, generator=generator),
        "group_dim": group_dim,
        "group_size": group_size,
        "num_bits": num_bits,
    }


def compute_ref(tensor, group_dim, group_size, num_bits):
    return kernelgenbench.baseline.v1_ds_grouped_asymmetric_quantize(
        tensor, group_dim, group_size, num_bits)


def compute_res(tensor, group_dim, group_size, num_bits):
    return kernelgenbench.solution.v1_ds_grouped_asymmetric_quantize(
        tensor, group_dim, group_size, num_bits)


def compare(ref_out, res_out, dtype=None):
    """Three outputs, identical uint8 payload and identical scale/min values."""
    assert isinstance(res_out, tuple) and len(res_out) == 3
    assert isinstance(ref_out, tuple) and len(ref_out) == 3
    quant_ref, scale_ref, min_ref = ref_out
    quant_res, scale_res, min_res = res_out
    assert quant_res.shape == quant_ref.shape
    assert quant_res.dtype == quant_ref.dtype == torch.uint8
    assert torch.equal(quant_res, quant_ref), "quantized bytes differ"
    for name, got, want in (("scale", scale_res, scale_ref),
                            ("min", min_res, min_ref)):
        assert got.shape == want.shape, f"{name} shape {got.shape} != {want.shape}"
        assert got.dtype == want.dtype, f"{name} dtype {got.dtype} != {want.dtype}"
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def compute_ref_performance(tensor, group_dim, group_size, num_bits):
    return checks.measure(
        kernelgenbench.baseline.v1_ds_grouped_asymmetric_quantize,
        {"tensor": tensor, "group_dim": group_dim, "group_size": group_size,
         "num_bits": num_bits},
        kernelgenbench.device,
    )


def compute_res_performance(tensor, group_dim, group_size, num_bits):
    return checks.measure(
        kernelgenbench.solution.v1_ds_grouped_asymmetric_quantize,
        {"tensor": tensor, "group_dim": group_dim, "group_size": group_size,
         "num_bits": num_bits},
        kernelgenbench.device,
    )


@label("v1_ds_grouped_asymmetric_quantize")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_ds_grouped_asymmetric_quantize(config):
    shape, group_dim, group_size, dtype, num_bits = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    for side in (ref_inputs, res_inputs):
        assert side["tensor"].shape == shape
        assert side["tensor"].dtype == dtype
        assert side["tensor"].shape[group_dim] % group_size == 0

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    # The implementation clones the input; the quantized bytes are a new buffer.
    for side in (ref_inputs, res_inputs):
        torch.testing.assert_close(side["tensor"].cpu(), inputs["tensor"], rtol=0, atol=0)
    assert not torch._C._is_alias_of(res_out[0], res_inputs["tensor"])

    return checks.timings(compute_ref_performance(**inputs),
                          compute_res_performance(**inputs))
