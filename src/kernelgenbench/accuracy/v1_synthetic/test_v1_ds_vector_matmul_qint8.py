"""Accuracy and performance for v1_ds_vector_matmul_qint8 (DeepSpeed INT8 GEMM).

Official sources (DeepSpeed pinned commit 8af59b2e7a91c227081617240e98aa282eadc5f9):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/transformer/inference/op_binding/vector_matmul.py#L12-L42
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/transformer/inference/csrc/pt_binding.cpp#L1331-L1380
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/transformer/inference/csrc/pt_binding.cpp#L1069-L1112

Contract (the ``q_int8`` mode of ``ds_vector_matmul<half>``): ``x`` is FP16
``[B, S, K]``, ``quantized_weight`` is INT8 ``[O, K]`` (transposed for the GEMM)
and ``quant_scale`` is FP32 ``[G]`` with ``O % G == 0``.  The weight is
dequantized per group as ``half(scale * code)``, then cuBLAS computes
``x @ dequant(weight).T`` in FP16 and returns a fresh FP16 ``[B, S, O]`` tensor.
This is a different contract from the dequantize+linear items: 3-D input,
INT8 weights with FP32 scales, no bias, FP16 only, and no ``quant_min``.  The
int8 entry needs the inference workspace pre-allocated for ``B*S*O`` elements;
the verification binding allocates it through the official
``allocate_workspace_fp16`` before the call.

Item: v1_ds_vector_matmul_qint8 (723-row 115, INT8 ``vector_matmul`` contract).
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows112_121_checks as checks


# (B, S, O, H, G): x [B, S, H] fp16, weight [O, H] int8, scale [G] fp32.
CASES = [
    (1, 1, 4, 4, 1),
    (2, 3, 12, 16, 3),
    (1, 7, 16, 64, 16),
    (2, 4, 32, 128, 4),
    (1, 2, 8, 4096, 2),
]

# Axes: one combo per case.
COMBOS = list(CASES)

TOLERANCE = (1e-3, 1e-3)


def input_build(config):
    """x [B, S, H] fp16; q [O, H] int8; scale [G] fp32 with O % G == 0."""
    batch, tokens, out_features, hidden, groups = config
    x = (((torch.arange(batch * tokens * hidden, dtype=torch.float32) * 7) % 31 - 15)
         .reshape(batch, tokens, hidden).half() / 16)
    q = (((torch.arange(out_features * hidden, dtype=torch.int32) * 19) % 256) - 128) \
        .to(torch.int8).reshape(out_features, hidden)
    scale = ((torch.arange(groups, dtype=torch.float32) % 7) + 1) / 128
    return dict(x=x, quantized_weight=q, quant_scale=scale,
                group_size=out_features // groups)


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_ds_vector_matmul_qint8(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_ds_vector_matmul_qint8(**inputs)


def compare(ref_out, res_out, batch, tokens, out_features):
    assert isinstance(res_out, torch.Tensor), f"expected a tensor, got {type(res_out).__name__}"
    assert res_out.shape == ref_out.shape == (batch, tokens, out_features)
    assert res_out.dtype == ref_out.dtype == torch.float16
    assert res_out.device == ref_out.device
    rtol, atol = TOLERANCE
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)


def compute_ref_performance(**inputs):
    return checks.measure(kernelgenbench.baseline.v1_ds_vector_matmul_qint8,
                          inputs, kernelgenbench.device)


def compute_res_performance(**inputs):
    return checks.measure(kernelgenbench.solution.v1_ds_vector_matmul_qint8,
                          inputs, kernelgenbench.device)


@label("v1_ds_vector_matmul_qint8")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_ds_vector_matmul_qint8(config):
    batch, tokens, out_features, hidden, groups = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    assert res_inputs["x"].shape == (batch, tokens, hidden)
    assert res_inputs["quantized_weight"].shape == (out_features, hidden)
    assert res_inputs["quant_scale"].shape == (groups,)
    assert all(res_inputs[name].is_contiguous()
               for name in ("x", "quantized_weight", "quant_scale"))

    before = {name: value.clone() for name, value in res_inputs.items()
              if isinstance(value, torch.Tensor)}
    guard = res_inputs["x"].clone()
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, batch, tokens, out_features)

    for name, value in before.items():
        assert torch.equal(value, res_inputs[name]), f"input {name} was modified"
        assert not torch._C._is_alias_of(res_out, res_inputs[name]), f"output aliases {name}"
    assert torch.equal(guard, res_inputs["x"])

    return checks.timings(compute_ref_performance(**inputs),
                          compute_res_performance(**inputs))
