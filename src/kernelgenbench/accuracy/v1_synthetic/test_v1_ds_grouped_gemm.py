"""Accuracy and performance for v1_ds_grouped_gemm.

Operator: DeepSpeed Triton grouped GEMM (2D x 3D), forward.

Official source (pinned commit 8af59b2e7a91c227081617240e98aa282eadc5f9):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/triton_ops/group_gemm_triton.py#L462-L489

Item: v1_ds_grouped_gemm (723-row 106).  ``compute_ref`` loads this file's
reference from ``kernelgenbench.baseline``; ``compute_res`` calls the official
Triton operator that the verifier bound for the same label.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows102_111_checks as checks


# (token counts per expert, K, N, trans_b, strided mat_a, dtype)
# Empty experts are legal.  strided=True keeps mat_a at stride 2 of a wider
# buffer; the official kernel compacts it internally.
COMBOS = [
    ((2, 0, 3), 7, 9, False, False, torch.float32),
    ((2, 0, 3), 7, 9, False, True, torch.float16),
    ((1, 3), 4, 5, True, True, torch.bfloat16),
    ((1, 3), 4, 5, True, False, torch.float32),
    ((0, 0), 4, 3, False, False, torch.float32),
    ((0, 2, 1), 8, 6, True, False, torch.float16),
    ((2, 1, 0), 5, 7, False, True, torch.bfloat16),
]

TOLERANCE = {
    torch.float32: (4e-3, 4e-3),
    torch.float16: (3e-2, 3e-2),
    torch.bfloat16: (3e-2, 3e-2),
}


def input_build(config):
    """mat_a [M, K], mat_b [E, K, N] or [E, N, K], offs [E] int32 cumulative."""
    counts, k, n, trans_b, strided, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(20261115)
    m = sum(counts)
    if strided:
        wide = torch.randn((m, k * 2), dtype=dtype, generator=generator)
        mat_a = wide[:, ::2]
    else:
        mat_a = torch.randn((m, k), dtype=dtype, generator=generator)
    shape = (len(counts), n, k) if trans_b else (len(counts), k, n)
    mat_b = torch.randn(shape, dtype=dtype, generator=generator) * 0.2
    offs = torch.tensor(counts, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    return {"mat_a": mat_a, "mat_b": mat_b, "offs": offs, "trans_b": trans_b}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_ds_grouped_gemm(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_ds_grouped_gemm(**inputs)


def compare(ref_out, res_out, dtype):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == dtype
    rtol, atol = TOLERANCE[dtype]
    if ref_out.numel():
        torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)


def compute_ref_performance(**inputs):
    return checks.measure(compute_ref, inputs, device=kernelgenbench.device)


def compute_res_performance(**inputs):
    return checks.measure(compute_res, inputs, device=kernelgenbench.device)


@label("v1_ds_grouped_gemm")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_ds_grouped_gemm(config):
    counts, k, n, trans_b, strided, dtype = config
    inputs = input_build(config)
    saved = checks.fresh(inputs)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    for side in (ref_inputs, res_inputs):
        assert side["mat_a"].shape == (sum(counts), k)
        assert side["mat_a"].is_contiguous() == (not strided)
        assert side["mat_b"].shape == ((len(counts), n, k) if trans_b
                                       else (len(counts), k, n))
        assert side["offs"].dtype == torch.int32

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)
    assert res_out.shape == (sum(counts), n)

    # No input is modified and the output is a fresh tensor.
    for side in (ref_inputs, res_inputs):
        for name in ("mat_a", "mat_b", "offs"):
            torch.testing.assert_close(side[name].cpu(), saved[name], rtol=0, atol=0)
        assert not torch._C._is_alias_of(res_out, side["mat_a"])
        assert not torch._C._is_alias_of(res_out, side["mat_b"])

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return checks.timings(ref_time, res_time)
