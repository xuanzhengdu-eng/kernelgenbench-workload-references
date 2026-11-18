"""Accuracy and performance for v1_sgl_mhc_mix_stats (SGLang Triton kernel).

Inputs are the same tensors consumed by SGLang's ``hc_mix_stats_sinkhorn``:
``x`` is [tokens, hc_mult, hidden], ``fn`` is [hc_mult3, hc_mult * hidden] and
together with ``scale``/``base`` is FP32.  Outputs are three FP32 tensors:
pre/post [tokens, hc_mult] and comb [tokens, hc_mult, hc_mult].  The operation
does not modify its inputs.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.activation_rows11_21_checks import (
    fresh, measure,
)
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# tokens, mHC streams, hidden, input dtype, Sinkhorn iterations
COMBOS = [
    (1, 4, 16, torch.float32, 1),
    (3, 4, 16, torch.bfloat16, 3),
    (2, 2, 32, torch.float32, 5),
    (2, 4, 16, torch.bfloat16, 1),
    (5, 4, 64, torch.bfloat16, 2),
]


def input_build(config):
    tokens, hc, hidden, dtype, iters = config
    generator = torch.Generator(device="cpu").manual_seed(1255)
    mix = (hc + 2) * hc
    return {
        "x": torch.randn((tokens, hc, hidden), dtype=dtype, device="cpu",
                         generator=generator),
        "fn": torch.randn((mix, hc * hidden), device="cpu", generator=generator) * 0.1,
        "scale": torch.tensor([0.5, 0.7, 0.9], dtype=torch.float32),
        "base": torch.randn((mix,), device="cpu", generator=generator) * 0.1,
        "hc_mult": hc,
        "sinkhorn_iters": iters,
        "rms_eps": 1e-6,
        "eps": 1e-6,
    }


def compute_ref(x, fn, scale, base, hc_mult, sinkhorn_iters, rms_eps, eps):
    return kernelgenbench.baseline.v1_sgl_mhc_mix_stats(
        x, fn, scale, base, hc_mult, sinkhorn_iters, rms_eps, eps)


def compute_res(x, fn, scale, base, hc_mult, sinkhorn_iters, rms_eps, eps):
    return kernelgenbench.solution.v1_sgl_mhc_mix_stats(
        x, fn, scale, base, hc_mult, sinkhorn_iters, rms_eps, eps)


def compare(ref_out, res_out, config):
    tokens, hc, _hidden, dtype, _iters = config
    assert isinstance(ref_out, tuple) and len(ref_out) == 3
    assert isinstance(res_out, tuple) and len(res_out) == 3
    for index, name in enumerate(("pre", "post", "comb")):
        expected, actual = ref_out[index], res_out[index]
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == expected.dtype == torch.float32
        assert actual.device == expected.device
        assert actual.shape == expected.shape
        expected_shape = (tokens, hc) if name != "comb" else (tokens, hc, hc)
        assert actual.shape == expected_shape, (name, actual.shape, expected_shape)
    # The fused reduction uses a different K-slice order from the reference GEMM.
    rtol, atol = (6e-4, 2e-5) if dtype == torch.float32 else (4e-3, 2e-4)
    for expected, actual in zip(ref_out, res_out):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_sgl_mhc_mix_stats")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_mhc_mix_stats(config):
    inputs = input_build(config)
    device_inputs = fresh(inputs, kernelgenbench.device)
    ref_inputs = fresh(device_inputs, kernelgenbench.device)
    res_inputs = fresh(device_inputs, kernelgenbench.device)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config)

    for name, before in device_inputs.items():
        if isinstance(before, torch.Tensor):
            torch.testing.assert_close(res_inputs[name], before, atol=0, rtol=0)
            torch.testing.assert_close(ref_inputs[name], before, atol=0, rtol=0)

    ref_time = compute_ref_performance(**device_inputs)
    res_time = compute_res_performance(**device_inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
