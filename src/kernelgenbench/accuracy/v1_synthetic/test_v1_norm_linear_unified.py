"""Accuracy and performance for v1_norm_linear_unified.

Operator: DeepSpeed QKV GEMM fallback (LayerNorm and RMSNorm variants), returning
``(projection, normalized_input)``.

Official source (pinned commit 8af59b2e7a91c227081617240e98aa282eadc5f9):
  qkv_gemm_fallback     qkv_gemm.py#L60-L67
  rms_qkv_gemm_fallback qkv_gemm.py#L70-L75

Item: v1_norm_linear_unified (723-row 107).  ``compute_ref`` loads this file's
reference from ``kernelgenbench.baseline``; ``compute_res`` calls the official
fallback selected by ``norm_type`` that the verifier bound for this label.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows102_111_checks as checks


# (norm_type, transpose, add_bias, dtype, strided_input)
# x is [2, 3, hidden]; the strided variant is a stride-2 view of a 2x-wide
# buffer.  weight is [out, hidden] when transpose=True, else [hidden, out].
COMBOS = [
    ("layer", True, True, torch.float32, False),
    ("layer", True, False, torch.float16, False),
    ("layer", False, True, torch.bfloat16, True),
    ("layer", False, False, torch.float32, True),
    ("layer", True, True, torch.bfloat16, False),
    ("rms", True, False, torch.float32, False),
    ("rms", True, False, torch.float16, True),
    ("rms", False, False, torch.bfloat16, False),
    ("rms", True, False, torch.bfloat16, True),
]

TOLERANCE = {
    torch.float32: (2e-5, 2e-5),
    torch.float16: (2e-3, 1e-3),
    torch.bfloat16: (1e-2, 8e-3),
}


def input_build(config):
    """x [2,3,hidden] (optionally strided), weight [out,hidden] or [hidden,out],
    gamma [hidden], beta [hidden] for layer norm, bias [out] when requested."""
    norm_type, transpose, add_bias, dtype, strided = config
    hidden, out_features = 32, 16
    generator = torch.Generator(device="cpu").manual_seed(20261115)
    if strided:
        wide = torch.randn((2, 3, hidden * 2), dtype=dtype, generator=generator)
        x = wide[..., ::2]
    else:
        x = torch.randn((2, 3, hidden), dtype=dtype, generator=generator)
    weight = torch.randn((out_features, hidden) if transpose else (hidden, out_features),
                         dtype=dtype, generator=generator)
    gamma = torch.randn((hidden,), dtype=dtype, generator=generator) * 0.1 + 1.0
    beta = (torch.randn((hidden,), dtype=dtype, generator=generator) * 0.1
            if norm_type == "layer" else None)
    bias = (torch.randn((out_features,), dtype=dtype, generator=generator) * 0.1
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
    return kernelgenbench.baseline.v1_norm_linear_unified(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_norm_linear_unified(**inputs)


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


@label("v1_norm_linear_unified")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_norm_linear_unified(config):
    norm_type, transpose, add_bias, dtype, strided = config
    inputs = input_build(config)
    saved = checks.fresh(inputs)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    for side in (ref_inputs, res_inputs):
        assert side["x"].shape == (2, 3, 32)
        assert side["x"].is_contiguous() == (not strided)
        assert side["weight"].shape == ((16, 32) if transpose else (32, 16))

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    # No input is modified and neither output aliases an input.
    for side in (ref_inputs, res_inputs):
        for name, value in side.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value.cpu(), saved[name], rtol=0, atol=0)
        for output in (ref_out if side is ref_inputs else res_out):
            for name, value in side.items():
                if isinstance(value, torch.Tensor):
                    assert not torch._C._is_alias_of(output, value), f"output aliases {name}"

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return checks.timings(ref_time, res_time)
