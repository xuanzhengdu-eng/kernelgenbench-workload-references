"""Accuracy and performance for v1_moe_row_weight_backward.

Contract: the DeepSpeed AutoEP row-weighting backward reached through autograd
on ``fused_row_weighting``. ``grad_rows`` equals ``grad_output.float() * weights``
cast to the row dtype; ``grad_weights`` equals the FP32 reduction
``sum(grad_output.float() * rows.float(), dim=1, keepdim=True)``. ``N == 0``
yields empty gradients (autograd may report None for an empty leaf) and
``H == 0`` yields zero weight gradients.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import moe_rows222_231_checks as checks


# (rows, hidden, strided grad_output, dtype)
COMBOS = [
    (n, h, strided, dtype)
    for n, h in [(0, 9), (3, 0), (1, 17), (4, 128), (3, 1027)]
    for dtype in (torch.float16, torch.bfloat16)
    for strided in (False, True)
]


def input_build(config):
    rows, hidden, strided, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(227)
    rows_t = torch.randn((rows, hidden), dtype=dtype, generator=generator)
    weights = torch.rand((rows, 1), dtype=torch.float32, generator=generator)
    grad = torch.randn((rows, hidden * (2 if strided else 1)), dtype=dtype, generator=generator)
    if strided:
        grad = grad[:, ::2]
    return {"rows": rows_t, "weights": weights, "grad_output": grad}


def compute_ref(rows, weights, grad_output):
    return kernelgenbench.baseline.v1_moe_row_weight_backward(rows, weights, grad_output)


def compute_res(rows, weights, grad_output):
    return kernelgenbench.solution.v1_moe_row_weight_backward(rows, weights, grad_output)


def compare(ref_out, res_out, dtype=None):
    assert isinstance(ref_out, tuple) and isinstance(res_out, tuple)
    assert len(ref_out) == len(res_out) == 2
    ref_rows, ref_weights = ref_out
    res_rows, res_weights = res_out
    assert res_rows.shape == ref_rows.shape and res_rows.dtype == ref_rows.dtype == dtype
    assert res_weights.shape == ref_weights.shape and res_weights.dtype == torch.float32
    rtol, atol = checks.tolerance(dtype)
    torch.testing.assert_close(res_rows, ref_rows, rtol=rtol, atol=atol)
    torch.testing.assert_close(res_weights, ref_weights, rtol=1e-5, atol=1e-5)


def compute_ref_performance(**inputs):
    return checks.measure(kernelgenbench.baseline.v1_moe_row_weight_backward, inputs, device=kernelgenbench.device)


def compute_res_performance(**inputs):
    return checks.measure(kernelgenbench.solution.v1_moe_row_weight_backward, inputs, device=kernelgenbench.device)


@label("v1_moe_row_weight_backward")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_moe_row_weight_backward(config):
    rows, hidden, strided, dtype = config
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

    # Explicit contract assertions, including the zero-hidden and empty cases.
    grad_rows_expected = (
        res_inputs["grad_output"].float() * res_inputs["weights"]
    ).to(dtype)
    grad_weights_expected = (
        res_inputs["grad_output"].float() * res_inputs["rows"].float()
    ).sum(dim=1, keepdim=True)
    torch.testing.assert_close(
        res_out[0], grad_rows_expected,
        rtol=checks.tolerance(dtype)[0], atol=checks.tolerance(dtype)[1],
    )
    torch.testing.assert_close(res_out[1], grad_weights_expected, rtol=1e-5, atol=1e-5)
    if hidden == 0:
        assert bool((res_out[1] == 0).all()), "empty hidden must yield zero weight grads"
    for value in res_out:
        assert value.dtype in (dtype, torch.float32)
    for name, value in before.items():
        assert torch.equal(res_inputs[name].cpu(), value), f"{name} was modified"

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return checks.timings(ref_time, res_time)
