"""Accuracy and performance for v1_router_linear_sglang.

Operator: SGLang ``UnquantizedLinearMethod.apply`` (unquantized linear fallback).

Official source (pinned commit 62384de1ae5e3bbd3159afd1639525b0bf39fbf9):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/quantization/unquant.py#L793-L839

Item: v1_router_linear_sglang (723-row 111), SGLang linear branch.  The official
method may dispatch to an AMX/AITER/BF16 GEMM backend; all backends share the
``F.linear`` contract checked here.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows102_111_checks as checks


# (input shape, dtype, bias_present)
# K is the last dimension; N = max(16, K // 2).
COMBOS = [
    ((2, 16), torch.float32, True),
    ((2, 16), torch.float16, False),
    ((2, 128), torch.bfloat16, True),
    ((2, 16, 128), torch.float32, False),
    ((2, 16, 128), torch.bfloat16, True),
    ((2, 4, 16, 128), torch.float16, True),
]

TOLERANCE = {
    torch.float32: (4e-6, 2e-6),
    torch.float16: (2e-3, 1e-3),
    torch.bfloat16: (1e-2, 8e-3),
}


def input_build(config):
    shape, dtype, bias_present = config
    out_features = max(16, shape[-1] // 2)
    generator = torch.Generator(device="cpu").manual_seed(20261115)
    inp = torch.randn(shape, dtype=dtype, generator=generator)
    weight = torch.randn((out_features, shape[-1]), dtype=dtype, generator=generator)
    bias = (torch.randn((out_features,), dtype=dtype, generator=generator)
            if bias_present else None)
    return {"inp": inp, "weight": weight, "bias": bias}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_router_linear_sglang(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_router_linear_sglang(**inputs)


def compare(ref_out, res_out, dtype):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == dtype
    rtol, atol = TOLERANCE[dtype]
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)


def compute_ref_performance(**inputs):
    return checks.measure(compute_ref, inputs, device=kernelgenbench.device)


def compute_res_performance(**inputs):
    return checks.measure(compute_res, inputs, device=kernelgenbench.device)


@label("v1_router_linear_sglang")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_router_linear_sglang(config):
    shape, dtype, bias_present = config
    out_features = max(16, shape[-1] // 2)
    inputs = input_build(config)
    saved = checks.fresh(inputs)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    for side in (ref_inputs, res_inputs):
        assert side["inp"].shape == shape
        assert side["weight"].shape == (out_features, shape[-1])
        if bias_present:
            assert side["bias"].shape == (out_features,)
        else:
            assert side["bias"] is None

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)
    assert res_out.shape == (*shape[:-1], out_features)

    for side in (ref_inputs, res_inputs):
        for name in ("inp", "weight"):
            torch.testing.assert_close(side[name].cpu(), saved[name], rtol=0, atol=0)
        if bias_present:
            torch.testing.assert_close(side["bias"].cpu(), saved["bias"], rtol=0, atol=0)
        assert not torch._C._is_alias_of(res_out, side["inp"])

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return checks.timings(ref_time, res_time)
