"""Accuracy and performance for v1_ds_lamb_update_ (row 157).

Official entry: DeepSpeed ``fused_lamb_cuda.lamb`` at commit
8af59b2e7a91c227081617240e98aa282eadc5f9, csrc/lamb/fused_lamb_cuda_kernel.cu
L184-L439.  The test checks the updated parameter, both optimizer moments, the
FP16 output copy, the returned LAMB coefficient, and the untouched gradient.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows_152_161_utils import (
    fresh,
    measure_stateful,
)
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (elements, grad dtype, copy output, eps mode, bias correction, all-zero state,
#  min_coeff, max_coeff)
COMBOS = [
    (33, torch.float32, False, 1, True, False, 0.1, 10.0),
    (1027, torch.float16, True, 0, False, False, 0.1, 10.0),
    (128, torch.float16, False, 1, True, False, 0.5, 1.5),
    (17, torch.float32, False, 0, True, True, 2.0, 5.0),
    (7, torch.float32, False, 1, False, False, 0.8, 0.9),
    (9, torch.float32, True, 1, False, False, 0.2, 2.0),
]

TOLERANCE = {
    torch.float32: (3e-4, 1e-6),
    torch.float16: (3e-3, 1e-3),
}


def input_build(config):
    """Build FP32 p/m/v, a grad in FP32 or FP16, and an optional FP16 p_copy."""
    n, grad_dtype, copy, mode, correction, zero, lo, hi = config
    rng = torch.Generator(device="cpu").manual_seed(15700 + n)
    p = torch.randn(n, generator=rng, dtype=torch.float32)
    g = torch.randn(n, generator=rng, dtype=grad_dtype)
    m = torch.randn(n, generator=rng, dtype=torch.float32) * 0.1
    v = torch.rand(n, generator=rng, dtype=torch.float32) * 0.2
    if zero:
        p.zero_()
        g.zero_()
        m.zero_()
        v.zero_()
    return {
        "p": p,
        "p_copy": torch.full((n if copy else 0,), -13.25, dtype=grad_dtype),
        "m": m,
        "v": v,
        "g": g,
        "lr": 0.003,
        "beta1": 0.9,
        "beta2": 0.999,
        "max_coeff": hi,
        "min_coeff": lo,
        "eps": 1e-5,
        "grad_scale": 2.0,
        "step": 7,
        "mode": mode,
        "bias_correction": int(correction),
        "decay": 0.01,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_ds_lamb_update_(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_ds_lamb_update_(**inputs)


def compare(ref_out, res_out, ref_inputs, res_inputs, grad_dtype):
    rtol, atol = TOLERANCE[grad_dtype]
    assert ref_out.shape == res_out.shape == (1,)
    assert ref_out.dtype == res_out.dtype == torch.float32
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)
    for name in ("p", "m", "v"):
        torch.testing.assert_close(res_inputs[name], ref_inputs[name], rtol=rtol, atol=atol)
    if grad_dtype == torch.float16 and ref_inputs["p_copy"].numel():
        torch.testing.assert_close(
            res_inputs["p_copy"], ref_inputs["p_copy"], rtol=3e-3, atol=1e-3
        )
    else:
        assert torch.equal(res_inputs["p_copy"], ref_inputs["p_copy"])
    assert torch.equal(res_inputs["g"], ref_inputs["g"])


def compute_ref_performance(inputs):
    return measure_stateful(compute_ref, inputs, kernelgenbench.device)


def compute_res_performance(inputs):
    return measure_stateful(compute_res, inputs, kernelgenbench.device)


@label("v1_ds_lamb_update_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_ds_lamb_update_(config):
    n, grad_dtype, copy, mode, correction, zero, lo, hi = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    for side in (ref_inputs, res_inputs):
        assert side["p"].shape == (n,) and side["p"].dtype == torch.float32
        assert side["g"].dtype == grad_dtype
        assert side["p_copy"].numel() == (n if copy else 0)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, ref_inputs, res_inputs, grad_dtype)

    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )
