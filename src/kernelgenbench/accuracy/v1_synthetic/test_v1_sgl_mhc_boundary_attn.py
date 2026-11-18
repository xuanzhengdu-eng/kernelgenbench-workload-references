"""Accuracy and performance for v1_sgl_mhc_boundary_attn.

x is the pending layer output (tokens, hidden) or None; residual is
(tokens, 4, hidden); post is (tokens, 4); comb is (tokens, 4, 4); pre_prev is
(tokens, 4) or None.  The branch matrix below covers both optional inputs.
The output is always a fresh BF16 tensor (tokens, hidden) after input RMSNorm.
The HIP fused kernel cannot run on this H800 node; the Solution binding uses
SGLang's torch post + Triton mhc_combine path, and the log records that.
"""
import pytest
import torch
import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.bias_activation_test_utils import fresh, measure
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (tokens, hidden, pending post present, previous pre present)
COMBOS = [
    (1, 64, False, False),
    (2, 64, False, True),
    (2, 64, True, False),
    (3, 128, True, True),
]


def input_build(config):
    tokens, hidden, has_post, has_pre = config
    g = torch.Generator(device="cpu").manual_seed(249000 + tokens * 100 + hidden)
    rand = lambda *shape: torch.randn(shape, generator=g, dtype=torch.float32)
    return {
        "x": ((rand(tokens, hidden) * 0.25).bfloat16() if has_post else None),
        "residual": (rand(tokens, 4, hidden) * 0.25).bfloat16(),
        "post": (rand(tokens, 4) * 0.3 if has_post else None),
        "comb": (rand(tokens, 4, 4) * 0.3 if has_post else None),
        "pre_prev": (rand(tokens, 4) * 0.3 if has_pre else None),
        "norm_weight": (rand(hidden) * 0.1 + 1.0).bfloat16(),
        "norm_eps": 1e-5,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_sgl_mhc_boundary_attn(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_sgl_mhc_boundary_attn(**inputs)


def compare(ref_out, res_out, config):
    tokens, hidden, _, _ = config
    assert res_out.shape == ref_out.shape == (tokens, hidden)
    assert res_out.dtype == ref_out.dtype == torch.bfloat16
    assert res_out.is_contiguous()
    torch.testing.assert_close(res_out.float(), ref_out.float(), rtol=2e-2, atol=2e-2)


def compute_ref_performance(inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(inputs):
    return measure(compute_res, inputs)


@label("v1_sgl_mhc_boundary_attn")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_mhc_boundary_attn(config):
    tokens, hidden, has_post, has_pre = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    assert (ref_inputs["x"] is None) == (not has_post)
    assert (ref_inputs["pre_prev"] is None) == (not has_pre)
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config)
    for side in (ref_inputs, res_inputs):
        assert not torch._C._is_alias_of(side["residual"], ref_out if side is ref_inputs else res_out)
        for name in ("residual", "norm_weight"):
            torch.testing.assert_close(side[name].cpu(), inputs[name], rtol=0, atol=0,
                                       equal_nan=True)
        if has_post:
            torch.testing.assert_close(side["post"].cpu(), inputs["post"], rtol=0, atol=0)
            torch.testing.assert_close(side["comb"].cpu(), inputs["comb"], rtol=0, atol=0)
        if has_pre:
            torch.testing.assert_close(side["pre_prev"].cpu(), inputs["pre_prev"],
                                       rtol=0, atol=0)
    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
