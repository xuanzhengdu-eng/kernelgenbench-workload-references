"""Accuracy and performance for v1_gather_extend_cache_locs_.

Official source (pinned commit 62384de1ae5e3bbd3159afd1639525b0bf39fbf9):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/speculative/cache_locs.py#L679-L749

Contract: allocate ``batch_size * draft_token_num`` int64 entries.  Each
request's ``req_to_token[req, start:end]`` range is copied to the running
prefix-sum output offset, so only the first ``sum(end - start)`` entries are
written.  The returned tensor is freshly allocated; the tail beyond that prefix
is untouched and is not compared.  ``batch_size`` may be zero.  Returns the
dense buffer; all inputs are read-only.

Item: v1_gather_extend_cache_locs_ (723-row 588, extend branch).
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows582_591_checks as checks


# req_pool_indices, start_offset, end_offset (variable lengths).
CASES = [
    ([2, 0, 1], [0, 3, 5], [2, 3, 7]),
    ([], [], []),
    ([1, 1], [2, 0], [2, 3]),
]
COMBOS = [(case, dtype) for case in CASES for dtype in (torch.int32, torch.int64)]


def input_build(config):
    """req_to_token int32 [3, 8]; start/end share one dtype; per-row lengths vary."""
    (req, start, end), dtype = config
    widths = [stop - begin for begin, stop in zip(start, end)]
    draft_token_num = max(widths + [1])
    return {
        "req_pool_indices": torch.tensor(req, dtype=dtype),
        "req_to_token": torch.arange(24, dtype=torch.int32).reshape(3, 8),
        "start_offset": torch.tensor(start, dtype=dtype),
        "end_offset": torch.tensor(end, dtype=dtype),
        "batch_size": len(req),
        "draft_token_num": draft_token_num,
        "device": torch.device("cpu"),
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_gather_extend_cache_locs_(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_gather_extend_cache_locs_(**inputs)


def compare(ref_out, res_out, total):
    """Both sides allocate the same int64 buffer; only the written prefix counts."""
    assert ref_out.shape == res_out.shape
    assert ref_out.dtype == res_out.dtype == torch.int64


def compute_ref_performance(**inputs):
    return checks.measure(compute_ref, inputs, inputs["req_to_token"].device)


def compute_res_performance(**inputs):
    return checks.measure(compute_res, inputs, inputs["req_to_token"].device)


@label("v1_gather_extend_cache_locs_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_gather_extend_cache_locs_(config):
    (req, start, end), dtype = config
    inputs = input_build(config)
    inputs["device"] = kernelgenbench.device
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    # Move the destination device to the selected one (fresh keeps tensors).
    ref_inputs["device"] = kernelgenbench.device
    res_inputs["device"] = kernelgenbench.device
    before = {key: value.clone() for key, value in res_inputs.items()
              if isinstance(value, torch.Tensor)}

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    total = sum(stop - begin for begin, stop in zip(start, end))
    expected_size = len(req) * inputs["draft_token_num"]
    assert ref_out.shape == res_out.shape == (expected_size,)
    compare(ref_out, res_out, total)

    expected = []
    for request, begin, stop in zip(req, start, end):
        expected.extend(inputs["req_to_token"][request, begin:stop].tolist())
    torch.testing.assert_close(res_out[:total],
                               torch.tensor(expected, dtype=torch.int64,
                                            device=res_out.device),
                               rtol=0, atol=0)
    for key, value in before.items():
        torch.testing.assert_close(ref_inputs[key], value, rtol=0, atol=0)
        torch.testing.assert_close(res_inputs[key], value, rtol=0, atol=0)

    ref_time = compute_ref_performance(**ref_inputs)
    res_time = compute_res_performance(**res_inputs)
    return checks.result(ref_time, res_time)
