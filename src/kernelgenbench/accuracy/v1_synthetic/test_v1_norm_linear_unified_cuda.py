"""Accuracy and performance for v1_norm_linear_unified_cuda.

Operator: DeepSpeed fused ``transformer_inference`` QKV GEMM (LayerNorm and
RMSNorm variants), the backend ``QKVGemmOp.forward`` selects on this build.

Official source (pinned commit 8af59b2e7a91c227081617240e98aa282eadc5f9):
  ds_qkv_gemm (CUDA binding) csrc/transformer/inference/csrc/pt_binding.cpp#L1031-L1066
  QKVGemmOp dispatch        deepspeed/ops/transformer/inference/op_binding/qkv_gemm.py#L60-L67 + #L70-L75

Item: v1_norm_linear_unified (723-row 107), CUDA backend reference.  The torch
fallback of the same entry is verified with its own reference/test.  The fused
kernel writes into DeepSpeed's inference workspace, so the binding copies the
returned tensors out before comparison.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows102_111_checks as checks


# (norm_type, transpose, add_bias, dtype)
# The fused kernel supports fp16 and bf16; weights follow the same
# [out, hidden] (transpose) / [hidden, out] layout as the fallback.
COMBOS = [
    ("layer", True, True, torch.float16),
    ("layer", True, False, torch.bfloat16),
    ("layer", False, True, torch.bfloat16),
    ("layer", False, False, torch.float16),
    ("rms", True, False, torch.float16),
    ("rms", True, False, torch.bfloat16),
    ("rms", False, False, torch.bfloat16),
]

# The fused RMSNorm kernel rounds inside the kernel (measured max abs diff vs
# the fallback: 0.0039 fp16, 0.0625 bf16 on outputs of magnitude ~20), so the
# allowance covers one kernel rounding step while still rejecting a dropped
# bias (bias is drawn at scale 0.5).
TOLERANCE = {
    torch.float16: (3e-3, 8e-3),
    torch.bfloat16: (2e-2, 9e-2),
}


def input_build(config):
    """x [2,3,16], weight [16,32] or [32,16], gamma [16], beta [16] for layer
    norm, bias [32] when requested."""
    norm_type, transpose, add_bias, dtype = config
    hidden, out_features = 16, 32
    generator = torch.Generator(device="cpu").manual_seed(20261115)
    x = torch.randn((2, 3, hidden), dtype=dtype, generator=generator)
    weight = torch.randn((out_features, hidden) if transpose else (hidden, out_features),
                         dtype=dtype, generator=generator) * 0.2
    gamma = torch.randn((hidden,), dtype=dtype, generator=generator) * 0.1 + 1.0
    beta = (torch.randn((hidden,), dtype=dtype, generator=generator) * 0.1
            if norm_type == "layer" else None)
    bias = (torch.randn((out_features,), dtype=dtype, generator=generator) * 0.5
            if add_bias else None)
    return {
        "x": x,
        "weight": weight,
        "gamma": gamma,
        "beta": beta,
        "bias": bias,
        "eps": 1e-5,
        "norm_type": norm_type,
        "transpose": transpose,
        "backend": "deepspeed",
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_norm_linear_unified_cuda(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_norm_linear_unified_cuda(**inputs)


def compare(ref_out, res_out, dtype):
    assert isinstance(ref_out, list) and isinstance(res_out, list)
    assert len(ref_out) == len(res_out) == 2
    rtol, atol = TOLERANCE[dtype]
    for expected, actual in zip(ref_out, res_out):
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype == dtype
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def compute_ref_performance(**inputs):
    return checks.measure(compute_ref, inputs, device=kernelgenbench.device)


def compute_res_performance(**inputs):
    return checks.measure(compute_res, inputs, device=kernelgenbench.device)


@label("v1_norm_linear_unified_cuda")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_norm_linear_unified_cuda(config):
    norm_type, transpose, add_bias, dtype = config
    inputs = input_build(config)
    saved = checks.fresh(inputs)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    for side in (ref_inputs, res_inputs):
        assert side["x"].shape == (2, 3, 16)
        assert side["weight"].shape == ((32, 16) if transpose else (16, 32))

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    for side in (ref_inputs, res_inputs):
        for name, value in side.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value.cpu(), saved[name], rtol=0, atol=0)
        for output in (ref_out if side is ref_inputs else res_out):
            assert not torch._C._is_alias_of(output, side["x"])

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return checks.timings(ref_time, res_time)
