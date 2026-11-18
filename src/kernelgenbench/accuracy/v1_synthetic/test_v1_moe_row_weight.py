"""Accuracy and performance for v1_moe_row_weight.

Contract: DeepSpeed ``fused_row_weighting``. Rows are contiguous fp16/bf16
``[N, H]`` and weights are contiguous fp32 ``[N, 1]``; the result is the eagerly
rounded ``(rows.float() * weights).to(rows.dtype)`` in a fresh tensor. No input
is mutated and empty dimensions are valid.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import moe_rows222_231_checks as checks


# (rows, hidden, dtype)
COMBOS = [
    (n, h, dtype)
    for n, h in [(0, 9), (3, 0), (1, 17), (4, 128), (3, 1027)]
    for dtype in (torch.float16, torch.bfloat16)
]


def input_build(config):
    rows, hidden, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(229)
    return {
        "rows": torch.randn((rows, hidden), dtype=dtype, generator=generator),
        "weights": torch.rand((rows, 1), dtype=torch.float32, generator=generator),
    }


def compute_ref(rows, weights):
    return kernelgenbench.baseline.v1_moe_row_weight(rows, weights)


def compute_res(rows, weights):
    return kernelgenbench.solution.v1_moe_row_weight(rows, weights)


def compare(ref_out, res_out, dtype=None):
    assert isinstance(ref_out, torch.Tensor) and isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape and res_out.dtype == ref_out.dtype == dtype
    rtol, atol = checks.tolerance(dtype)
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)


def compute_ref_performance(**inputs):
    return checks.measure(kernelgenbench.baseline.v1_moe_row_weight, inputs, device=kernelgenbench.device)


def compute_res_performance(**inputs):
    return checks.measure(kernelgenbench.solution.v1_moe_row_weight, inputs, device=kernelgenbench.device)


@label("v1_moe_row_weight")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_moe_row_weight(config):
    rows, hidden, dtype = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    before = {
        name: value.cpu().clone()
        for name, value in res_inputs.items()
        if isinstance(value, torch.Tensor)
    }

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    expected = (res_inputs["rows"].float() * res_inputs["weights"]).to(dtype)
    torch.testing.assert_close(
        res_out, expected,
        rtol=checks.tolerance(dtype)[0], atol=checks.tolerance(dtype)[1],
    )
    assert res_out.is_contiguous()
    if res_out.numel() and res_inputs["rows"].numel():
        assert res_out.data_ptr() != res_inputs["rows"].data_ptr()
    for name, value in before.items():
        assert torch.equal(res_inputs[name].cpu(), value), f"{name} was modified"

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return checks.timings(ref_time, res_time)
