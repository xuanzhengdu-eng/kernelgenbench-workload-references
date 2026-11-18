"""Accuracy and performance for v1_ds_grouped_gemm_backward.

Operator: explicit DeepSpeed grouped GEMM backward (the autograd backward of
``group_gemm_triton``), including optional input/weight gradients.

Official source (pinned commit 8af59b2e7a91c227081617240e98aa282eadc5f9):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/triton_ops/group_gemm_triton.py#L430-L459

Item: v1_ds_grouped_gemm_backward (723-row 105).  ``compute_ref`` loads this
file's reference from ``kernelgenbench.baseline``; ``compute_res`` runs the
official forward under autograd and takes ``torch.autograd.grad`` so the real
``_group_gemm_dw_kernel`` / ``_group_gemm_kernel`` backward path executes.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows102_111_checks as checks


# (token counts per expert, K, N, trans_b, strided mat_a, need_grad_a,
#  need_grad_b, dtype)
COMBOS = [
    ((2, 0, 3), 7, 9, False, False, True, True, torch.float32),
    ((1, 3), 4, 5, True, True, True, True, torch.float16),
    ((2, 1, 0), 8, 6, True, False, True, False, torch.float32),
    ((1, 2), 5, 7, False, True, False, True, torch.bfloat16),
    ((0, 2, 1), 4, 5, False, False, True, True, torch.float32),
    ((3, 1), 6, 4, True, False, True, True, torch.bfloat16),
]

TOLERANCE = {
    torch.float32: (5e-3, 5e-3),
    torch.float16: (3e-2, 3e-2),
    torch.bfloat16: (4e-2, 4e-2),
}


def input_build(config):
    """mat_a [M,K], mat_b [E,K,N] or [E,N,K], offs [E] int32 cumulative,
    grad_out [M,N]."""
    counts, k, n, trans_b, strided, need_grad_a, need_grad_b, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(20261115)
    m = sum(counts)
    if strided:
        wide = torch.randn((m, k * 2), dtype=dtype, generator=generator)
        mat_a = wide[:, ::2]
    else:
        mat_a = torch.randn((m, k), dtype=dtype, generator=generator)
    shape = (len(counts), n, k) if trans_b else (len(counts), k, n)
    mat_b = torch.randn(shape, dtype=dtype, generator=generator) * 0.2
    grad_out = torch.randn((m, n), dtype=dtype, generator=generator)
    offs = torch.tensor(counts, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    return {
        "mat_a": mat_a,
        "mat_b": mat_b,
        "offs": offs,
        "grad_out": grad_out,
        "trans_b": trans_b,
        "need_grad_a": need_grad_a,
        "need_grad_b": need_grad_b,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_ds_grouped_gemm_backward(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_ds_grouped_gemm_backward(**inputs)


def compare(ref_out, res_out, dtype):
    assert isinstance(ref_out, tuple) and isinstance(res_out, tuple)
    assert len(ref_out) == len(res_out) == 4
    rtol, atol = TOLERANCE[dtype]
    for index, (expected, actual) in enumerate(zip(ref_out, res_out)):
        if expected is None or actual is None:
            assert expected is None and actual is None, f"tuple item {index} is not both None"
            continue
        assert actual.shape == expected.shape, f"tuple item {index} shape mismatch"
        assert actual.dtype == expected.dtype == dtype, f"tuple item {index} dtype mismatch"
        if expected.numel():
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def compute_ref_performance(**inputs):
    return checks.measure(compute_ref, inputs, device=kernelgenbench.device)


def compute_res_performance(**inputs):
    return checks.measure(compute_res, inputs, device=kernelgenbench.device)


@label("v1_ds_grouped_gemm_backward")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_ds_grouped_gemm_backward(config):
    counts, k, n, trans_b, strided, need_grad_a, need_grad_b, dtype = config
    inputs = input_build(config)
    saved = checks.fresh(inputs)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    for side in (ref_inputs, res_inputs):
        assert side["mat_a"].shape == (sum(counts), k)
        assert side["mat_b"].shape == ((len(counts), n, k) if trans_b
                                       else (len(counts), k, n))
        assert side["grad_out"].shape == (sum(counts), n)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    # Optional-gradient contract: disabled gradients are None in both tuples.
    assert (ref_out[0] is None) == (not need_grad_a)
    assert (res_out[0] is None) == (not need_grad_a)
    assert (ref_out[1] is None) == (not need_grad_b)
    assert (res_out[1] is None) == (not need_grad_b)
    assert ref_out[2] is None and ref_out[3] is None
    assert res_out[2] is None and res_out[3] is None
    if need_grad_a:
        assert ref_out[0].shape == (sum(counts), k)
    if need_grad_b:
        assert ref_out[1].shape == ((len(counts), n, k) if trans_b
                                    else (len(counts), k, n))

    for side in (ref_inputs, res_inputs):
        for name in ("mat_a", "mat_b", "grad_out"):
            torch.testing.assert_close(side[name].cpu(), saved[name], rtol=0, atol=0)
        for grad in (side is res_inputs and res_out or ref_out)[:2]:
            if grad is not None:
                assert not torch._C._is_alias_of(grad, side["mat_a"])
                assert not torch._C._is_alias_of(grad, side["mat_b"])

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return checks.timings(ref_time, res_time)
