"""Accuracy and performance for v1_filter_top_k_top_p_triton.

Official source (pinned commit f358f0578a83f4e1fe6fc49a492d3ae9362de8e5):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/sample/ops/topk_topp_triton.py#L1618-L1710

Contract: float32 ``[B, V]`` contiguous logits.  The Triton kernel ranks values
and keeps exactly ``k`` entries per row; a value tied with the k-th score is
therefore dropped when another equal value already occupies the k-th rank,
unlike the PyTorch fallback's ``< threshold`` cut.  Top-p is then applied to
the survivors by cumulative probability.  The returned tensor may alias or
replace the input, so the test does not require in-place mutation.  Strided
rows are not part of this item (the kernel makes them contiguous first).

Item: v1_filter_top_k_top_p_ (723-row 586, Triton backend).
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows582_591_checks as checks


# Shape, mode (none/k/p/both), and whether row 0 has two equal 3.0 logits.
COMBOS = [
    (shape, mode, tie)
    for shape in ((3, 13), (3, 128))
    for mode in ("none", "k", "p", "both")
    for tie in (False, True)
]


def input_build(config):
    """logits float32 ``[B, V]`` contiguous; k int64 [B], p float32 [B]."""
    (batch, vocab), mode, tie = config
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    logits = torch.randn((batch, vocab), dtype=torch.float32, generator=generator)
    if tie:
        logits[0, 0] = logits[0, 1] = 3.
    return {
        "logits": logits,
        "k": torch.tensor([1, max(1, vocab // 2), vocab], dtype=torch.int64)
        if mode in ("k", "both") else None,
        "p": torch.tensor([1., .9, .5], dtype=torch.float32)
        if mode in ("p", "both") else None,
    }


def compute_ref(logits, k=None, p=None):
    return kernelgenbench.baseline.v1_filter_top_k_top_p_triton(logits, k, p)


def compute_res(logits, k=None, p=None):
    return kernelgenbench.solution.v1_filter_top_k_top_p_triton(logits, k, p)


def compare(ref_out, res_out, dtype, tie):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == dtype
    if not tie:
        torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0)
        return
    # Tied logits are interchangeable for sampling; compare the surviving
    # multiset row by row instead of demanding the same tied column.
    for row in range(ref_out.shape[0]):
        ref_values = ref_out[row][torch.isfinite(ref_out[row])].sort().values
        res_values = res_out[row][torch.isfinite(res_out[row])].sort().values
        assert ref_values.numel() == res_values.numel()
        torch.testing.assert_close(res_values, ref_values, rtol=0, atol=0)


def compute_ref_performance(logits, k=None, p=None):
    initial = logits.clone()

    def call():
        logits.copy_(initial)
        return compute_ref(logits, k, p)

    return checks.measure(call, {}, logits.device)


def compute_res_performance(logits, k=None, p=None):
    initial = logits.clone()

    def call():
        logits.copy_(initial)
        return compute_res(logits, k, p)

    return checks.measure(call, {}, logits.device)


@label("v1_filter_top_k_top_p_triton")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_filter_top_k_top_p_triton(config):
    shape, mode, tie = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    k_before = None if ref_inputs["k"] is None else ref_inputs["k"].clone()
    p_before = None if ref_inputs["p"] is None else ref_inputs["p"].clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, torch.float32, tie)
    if k_before is not None:
        torch.testing.assert_close(res_inputs["k"], k_before, rtol=0, atol=0)
    if p_before is not None:
        torch.testing.assert_close(res_inputs["p"], p_before, rtol=0, atol=0)

    if mode in ("k", "both"):
        # Strict rank cut: exactly k finite entries per row before top-p.
        for row, k in enumerate(inputs["k"].tolist()):
            finite = torch.count_nonzero(torch.isfinite(res_out[row]))
            if mode == "k":
                assert finite == k
            else:
                assert finite <= k
    if tie and mode == "k":
        assert torch.count_nonzero(torch.isfinite(res_out[0])) == 1
    if tie and mode == "none":
        assert torch.count_nonzero(torch.isfinite(res_out[0])) == shape[1]

    ref_time = compute_ref_performance(**ref_inputs)
    res_time = compute_res_performance(**res_inputs)
    return checks.result(ref_time, res_time)
