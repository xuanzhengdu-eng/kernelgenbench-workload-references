"""Accuracy and performance for v1_sgl_mhc_combine_norm (SGLang mHC combine).

streams is (tokens, 4, hidden), pre is (tokens, 4) or None; the output is the
pre-weighted sum followed by RMSNorm.  For BF16 / 4 streams / hidden=5120 the
official fused hc_combine_norm Triton kernel is reached; other widths use the
mhc_combine Triton kernel plus F.rms_norm.  The candidate solution dispatches
exactly like deepseek_v4_mhc._combine.
"""
import pytest
import torch
import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.bias_activation_test_utils import fresh, measure
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (tokens, hidden, has_pre)
COMBOS = [
    (2, 5120, True),
    (4, 5120, True),
    (1, 5120, True),
    (3, 128, True),
    (2, 256, False),
]


def input_build(config):
    tokens, hidden, has_pre = config
    g = torch.Generator(device="cpu").manual_seed(247000 + tokens * 100 + hidden)
    return {
        "streams": torch.randn((tokens, 4, hidden), dtype=torch.bfloat16, generator=g),
        "pre": (torch.randn((tokens, 4), dtype=torch.float32, generator=g)
                if has_pre else None),
        "norm_weight": (torch.randn((hidden,), dtype=torch.bfloat16, generator=g)
                        * 0.1 + 1.0),
        "eps": 1e-6,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_sgl_mhc_combine_norm(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_sgl_mhc_combine_norm(**inputs)


def compare(ref_out, res_out, config):
    tokens, hidden, _ = config
    assert res_out.shape == ref_out.shape == (tokens, hidden)
    assert res_out.dtype == ref_out.dtype == torch.bfloat16
    torch.testing.assert_close(res_out.float(), ref_out.float(), rtol=2e-2, atol=2e-2)


def compute_ref_performance(inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(inputs):
    return measure(compute_res, inputs)


@label("v1_sgl_mhc_combine_norm")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_mhc_combine_norm(config):
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config)
    for side in (ref_inputs, res_inputs):
        for name in ("streams", "norm_weight"):
            torch.testing.assert_close(side[name].cpu(), inputs[name], rtol=0, atol=0,
                                       equal_nan=True)
        if inputs["pre"] is not None:
            torch.testing.assert_close(side["pre"].cpu(), inputs["pre"], rtol=0,
                                       atol=0, equal_nan=True)
    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
