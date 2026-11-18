"""Accuracy and performance for v1_sgl_renorm_top_k (FlashInfer fast-kernel branch).

Official source (pinned commit 62384de1ae5e3bbd3159afd1639525b0bf39fbf9):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/sampling_renorm.py#L73-L84

Contract: ``probs`` is nonnegative ``[B, V]``; ``top_ks`` is clamped to
``[1, V]``.  The largest ``max(top_ks)`` entries are taken per row, entries at
rank ``>= k`` are zeroed, survivors are renormalized and scattered back to the
original columns.  The output is freshly allocated; the input, including a
strided input buffer, is unchanged.  This branch calls the platform fast kernel
(FlashInfer ``top_k_renorm_probs`` on CUDA) with ``deterministic=False``.  That
kernel requires contiguous rows, so the fast item only uses contiguous inputs;
the strided contract stays with the torch-fallback item.

Item: v1_sgl_renorm_top_k (723-row 585, top-k branch, FlashInfer fast kernel).
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows582_591_checks as checks


# B, V, per-row top-k (0 and >V exercise the clamp), strided input.
CASES = [(3, 8, (1, 3, 8), False), (3, 17, (1, 4, 30), False)]
COMBOS = [(case, dtype) for case in CASES
          for dtype in (torch.float32, torch.float16, torch.bfloat16)]

TOLERANCE = {
    torch.float32: (1e-6, 1e-7),
    torch.float16: (2e-3, 1e-3),
    torch.bfloat16: (1e-2, 8e-3),
}


def input_build(config):
    """probs ``[B, V]`` normalized rows; top_ks int32 ``[B]``; optional strided rows."""
    (batch, vocab, limits, strided), dtype = config
    generator = torch.Generator(device="cpu").manual_seed(20261110)
    probs = torch.rand((batch, vocab), dtype=dtype, generator=generator)
    probs = probs / probs.sum(-1, keepdim=True)
    if strided:
        storage = torch.zeros((batch, vocab + 3), dtype=dtype)
        storage[:, :vocab] = probs
        probs = storage[:, :vocab]
    return {"probs": probs, "top_ks": torch.tensor(limits, dtype=torch.int32)}


def compute_ref(probs, top_ks):
    return kernelgenbench.baseline.v1_sgl_renorm_top_k_fast(probs, top_ks)


def compute_res(probs, top_ks):
    return kernelgenbench.solution.v1_sgl_renorm_top_k_fast(probs, top_ks)


def compare(ref_out, res_out, dtype):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == dtype
    rtol, atol = TOLERANCE[dtype]
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)


def compute_ref_performance(probs, top_ks):
    return checks.measure(compute_ref, {"probs": probs, "top_ks": top_ks}, probs.device)


def compute_res_performance(probs, top_ks):
    return checks.measure(compute_res, {"probs": probs, "top_ks": top_ks}, probs.device)


@label("v1_sgl_renorm_top_k_fast")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_renorm_top_k_fast(config):
    (batch, vocab, limits, strided), dtype = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    before = ref_inputs["probs"].clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    assert res_out.shape == (batch, vocab)
    assert not torch._C._is_alias_of(res_out, res_inputs["probs"])
    torch.testing.assert_close(res_out.sum(-1), torch.ones(batch, dtype=dtype,
                                                          device=res_out.device),
                               rtol=2e-2 if dtype is torch.bfloat16 else 2e-3,
                               atol=2e-2 if dtype is torch.bfloat16 else 1e-3)
    torch.testing.assert_close(ref_inputs["probs"], before, rtol=0, atol=0)
    torch.testing.assert_close(res_inputs["probs"], before, rtol=0, atol=0)
    assert res_inputs["probs"].stride() == ref_inputs["probs"].stride()
    # Clamping: k=0 behaves like k=1, k>V keeps the full row.
    clamped = [max(1, min(int(k), vocab)) for k in limits]
    for row, k in enumerate(clamped):
        assert torch.count_nonzero(res_out[row]) == k

    ref_time = compute_ref_performance(**ref_inputs)
    res_time = compute_res_performance(**res_inputs)
    return checks.result(ref_time, res_time)
