"""Accuracy and performance for v1_moe_sinkhorn (row 68).

Contract: Megatron MoE Sinkhorn exponentiates the cost matrix and alternates row
and column scaling until the mean change of d1 falls below ``tol``.  The output
is d1 * exp(cost) * d0[:, None]; its row and column sums settle at 1/num_rows.
Official: https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/moe_utils.py#L181-L203
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows62_71_checks import (
    fresh, measure, result, tolerance, unchanged,
)
from sandbox.verifier.test_parametrize import label, parametrize

# (rows, experts, tol, dtype); the official helper is dtype-preserving.
COMBOS = [
    (rows, experts, tol, dtype)
    for rows, experts in [(16, 16), (128, 16), (5, 4), (1, 8)]
    for tol in (1e-4, 1e-6)
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
]


def input_build(config):
    rows, experts, tol, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(6801)
    cost = torch.randn((rows, experts), dtype=dtype, generator=generator) * 0.5
    return {"cost": cost, "tol": tol}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_moe_sinkhorn(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_moe_sinkhorn(**inputs)


def compare(ref_out, res_out, dtype):
    assert isinstance(res_out, torch.Tensor) and isinstance(ref_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == dtype
    rtol, atol = tolerance(dtype)
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)


@label("v1_moe_sinkhorn")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_moe_sinkhorn(config):
    rows, experts, tol, dtype = config
    inputs = input_build(config)
    before = {k: v.clone() for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    assert ref_inputs["cost"].shape == (rows, experts) and ref_inputs["cost"].dtype == dtype

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)
    unchanged(ref_inputs, before)
    unchanged(res_inputs, before)

    # Contract: the tolerance-driven scaling drives the row marginals to
    # 1/num_rows and the column marginals to 1/num_experts (each update divides
    # by the length of the opposite dimension).  A fixed small number of
    # iterations leaves the marginals off, so this is a real assertion.
    margin = 5e-3 if dtype == torch.float32 else 4e-2
    torch.testing.assert_close(res_out.float().sum(dim=-1),
                               torch.full((rows,), 1.0 / rows, device=res_out.device),
                               rtol=0, atol=margin)
    torch.testing.assert_close(res_out.float().sum(dim=0),
                               torch.full((experts,), 1.0 / experts, device=res_out.device),
                               rtol=0, atol=margin)
    assert torch.isfinite(res_out).all()

    ref_time = measure(lambda: compute_ref(**ref_inputs))
    res_time = measure(lambda: compute_res(**res_inputs))
    return result(ref_time, res_time)
