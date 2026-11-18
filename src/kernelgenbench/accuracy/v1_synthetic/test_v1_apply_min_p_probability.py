"""Accuracy and performance for v1_apply_min_p_probability.

This is vLLM's standalone logits-processor backend (``MinPLogitsProcessor.apply``)
with a different contract from the GPU log-scale kernel: it filters on softmax
probabilities, and it returns the same logits object instead of ``None``. Rows
mapped to min_p 0 keep every entry because the threshold product is 0.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.sampling_test_utils import (
    assert_unchanged,
    measure_inplace,
    move,
)
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

SHAPES = [(6, 13), (6, 2050)]
COMBOS = [
    (shape, dtype, row_padding)
    for shape in SHAPES
    for dtype in (torch.float32, torch.bfloat16)
    for row_padding in (False, True)
]


def input_build(config):
    (rows, vocab), dtype, row_padding = config
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    base = torch.randn((rows, vocab + (8 if row_padding else 0)), generator=generator).to(dtype)
    logits = base[:, :vocab]
    return {
        "logits": logits,
        "expanded_idx_mapping": torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int64),
        "min_p": torch.tensor([0.0, 0.25, 1.0], dtype=torch.float32),
    }


def compute_ref(logits, **kwargs):
    return kernelgenbench.baseline.v1_apply_min_p_probability(logits, **kwargs)


def compute_res(logits, **kwargs):
    return kernelgenbench.solution.v1_apply_min_p_probability(logits, **kwargs)


def compute_ref_performance(initial, logits, **kwargs):
    return measure_inplace(compute_ref, {"logits": logits, **kwargs}, initial, logits.device)


def compute_res_performance(initial, logits, **kwargs):
    return measure_inplace(compute_res, {"logits": logits, **kwargs}, initial, logits.device)


@label("v1_apply_min_p_probability")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_apply_min_p_probability(config):
    (rows, vocab), dtype, row_padding = config
    inputs = input_build(config)
    device = kernelgenbench.device
    ref_inputs = move(inputs, device)
    res_inputs = move(inputs, device)
    initial = res_inputs["logits"].clone()
    min_p = res_inputs["min_p"].clone()
    mapping = res_inputs["expanded_idx_mapping"].clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    # This backend returns the logits object, unlike the log-scale kernel.
    assert ref_out is ref_inputs["logits"]
    assert res_out is res_inputs["logits"]
    out = res_inputs["logits"]
    assert out.stride(0) == vocab + (8 if row_padding else 0)
    assert_unchanged(
        {k: v for k, v in res_inputs.items() if k != "logits"},
        {"min_p": min_p, "expanded_idx_mapping": mapping},
    )

    # The surviving entries are those with probability >= max_prob * min_p.
    for row in range(initial.shape[0]):
        p = float(min_p[mapping[row]])
        probs = torch.softmax(initial[row].float(), dim=-1)
        threshold = float(probs.max()) * p
        keep = probs >= threshold
        assert torch.isneginf(out[row][~keep]).all()
        assert torch.isfinite(out[row][keep]).all()
        torch.testing.assert_close(
            out[row][keep].float(), initial[row][keep].float(), rtol=0, atol=0
        )
        if p == 0:
            assert bool(keep.all())

    rtol, atol = {
        torch.float32: (1e-6, 1e-6),
        torch.bfloat16: (1e-2, 8e-3),
    }[dtype]
    torch.testing.assert_close(
        out, ref_inputs["logits"], rtol=rtol, atol=atol, equal_nan=True
    )

    perf_ref = move(inputs, device)
    perf_res = move(inputs, device)
    ref_time = compute_ref_performance(perf_ref["logits"].clone(), **perf_ref)
    res_time = compute_res_performance(perf_res["logits"].clone(), **perf_res)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )
