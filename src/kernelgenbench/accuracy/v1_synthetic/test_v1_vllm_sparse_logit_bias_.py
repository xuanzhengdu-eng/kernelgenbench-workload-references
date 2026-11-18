"""Accuracy and performance for v1_vllm_sparse_logit_bias_.

The bias is applied through vLLM's real ``LogitBiasLogitsProcessor`` state:
request/token pairs and a float32 bias vector are staged exactly as
``update_state`` does, then ``apply`` mutates the logits. Cases cover multiple
pairs per row, a single pair, an empty (disabled) list, a 1x1 vocabulary and a
non-contiguous row stride.
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

# batch, vocab, selected (row, token) pairs, enabled, non-contiguous row stride.
CASES = [
    (3, 8, ((0, 1), (0, 3), (2, 5)), True, False),
    (2, 7, ((1, 6),), True, True),
    (2, 8, (), False, True),
    (1, 1, ((0, 0),), True, False),
    (4, 33, ((0, 0), (1, 32), (3, 16), (3, 16 - 1)), True, False),
]
COMBOS = [
    (case, dtype)
    for case in CASES
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
]


def input_build(config):
    (batch, vocab, indices, enabled, strided), dtype = config
    generator = torch.Generator(device="cpu").manual_seed(20261110)
    logits = torch.randn((batch, vocab), dtype=dtype, generator=generator)
    if strided:
        base = torch.empty((batch, 2 * vocab), dtype=dtype)
        base[:, :vocab] = logits
        logits = base[:, :vocab]
    return {
        "logits": logits,
        "rows": torch.tensor([row for row, _ in indices], dtype=torch.int32),
        "tokens": torch.tensor([token for _, token in indices], dtype=torch.int32),
        "bias": torch.linspace(-0.75, 1.25, len(indices), dtype=torch.float32),
        "enabled": enabled,
    }


def compute_ref(logits, **kwargs):
    return kernelgenbench.baseline.v1_vllm_sparse_logit_bias_(logits, **kwargs)


def compute_res(logits, **kwargs):
    return kernelgenbench.solution.v1_vllm_sparse_logit_bias_(logits, **kwargs)


def compare(ref_out, res_out, dtype):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.dtype == ref_out.dtype == dtype
    assert res_out.shape == ref_out.shape
    torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0)


def compute_ref_performance(initial, logits, **kwargs):
    return measure_inplace(compute_ref, {"logits": logits, **kwargs}, initial, logits.device)


def compute_res_performance(initial, logits, **kwargs):
    return measure_inplace(compute_res, {"logits": logits, **kwargs}, initial, logits.device)


@label("v1_vllm_sparse_logit_bias_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_sparse_logit_bias_(config):
    (batch, vocab, indices, enabled, strided), dtype = config
    inputs = input_build(config)
    device = kernelgenbench.device
    ref_inputs = move(inputs, device)
    res_inputs = move(inputs, device)
    initial = res_inputs["logits"].clone()
    metadata = {
        k: v.clone() for k, v in res_inputs.items() if isinstance(v, torch.Tensor) and k != "logits"
    }

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)

    # The processor updates the caller's tensor and returns that same object.
    assert ref_out is ref_inputs["logits"]
    assert res_out is res_inputs["logits"]
    compare(ref_out, res_out, dtype)
    assert res_inputs["logits"].stride(0) == (2 * vocab if strided else vocab)

    # Only the selected entries move, and exactly by their staged bias.
    expected = initial.clone()
    if enabled:
        for index, (row, token) in enumerate(indices):
            expected[row, token] += res_inputs["bias"][index]
    torch.testing.assert_close(res_out, expected, rtol=0, atol=1e-6)
    if not enabled:
        assert torch.equal(res_out, initial)
    assert_unchanged(
        {k: v for k, v in res_inputs.items() if k != "logits"}, metadata
    )

    perf_ref = move(inputs, device)
    perf_res = move(inputs, device)
    ref_time = compute_ref_performance(perf_ref["logits"].clone(), **perf_ref)
    res_time = compute_res_performance(perf_res["logits"].clone(), **perf_res)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )
