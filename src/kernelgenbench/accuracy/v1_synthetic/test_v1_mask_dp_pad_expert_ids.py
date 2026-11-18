"""Accuracy and performance for v1_mask_dp_pad_expert_ids (SGLang DP mask).

Official entry: SGLang ``mask_dp_pad_moe_topk_ids``, pinned commit
62384de1ae5e3bbd3159afd1639525b0bf39fbf9, source lines 808-853.  The upstream
function reads ``_DpGatheredBufferWrapper`` state; the candidate binding sets
that state from this test's explicit ``counts``/``max_len`` and then calls the
official Triton kernel.

Input: ``topk_ids`` [rows, top_k] with rows == len(counts) * max_len for the
active path.  Padding rows are determined by ``r % max_len >= counts[r //
max_len]`` and are set to -1 in place; real rows are untouched.  Mismatch,
``counts=None`` and ``max_len<=0`` are no-ops.  Returns None.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic.rows202_211_test_utils import fresh, measure, result


# rows, max_len, top_k, counts, dtype.  Rows deliberately differ from
# len(counts)*max_len in the "mismatch" entries.
COMBOS = [
    (8, 4, 3, [0, 4], torch.int32),
    (8, 4, 3, [0, 4], torch.int64),
    (24, 8, 4, [2, 8, 3], torch.int32),
    (20, 5, 2, [5, 5, 5, 5], torch.int32),
    (8, 4, 3, [0, 0], torch.int64),
    (9, 4, 3, [4, 4], torch.int32),      # layout mismatch
    (8, 4, 3, None, torch.int32),        # no counts
    (8, 0, 3, [4, 4], torch.int32),      # zero max_len
]


def input_build(config):
    """Values 0..12 make a masked -1 unambiguous."""
    rows, max_len, top_k, counts, dtype = config
    topk_ids = (torch.arange(rows * top_k, dtype=dtype) % 13).reshape(rows, top_k)
    count_tensor = None if counts is None else torch.tensor(counts, dtype=torch.int32)
    return {"topk_ids": topk_ids, "counts": count_tensor, "max_len": max_len}


def compute_ref(topk_ids, counts, max_len):
    return kernelgenbench.baseline.v1_mask_dp_pad_expert_ids(topk_ids, counts, max_len)


def compute_res(topk_ids, counts, max_len):
    return kernelgenbench.solution.v1_mask_dp_pad_expert_ids(topk_ids, counts, max_len)


def _expected(topk_ids, counts, max_len):
    expected = topk_ids.clone()
    if counts is None or max_len <= 0 or topk_ids.shape[0] != counts.numel() * max_len:
        return expected
    rows = torch.arange(topk_ids.shape[0], device=topk_ids.device)
    pad = rows % max_len >= counts[rows // max_len]
    expected[pad] = -1
    return expected


def compare(ref_out, res_out):
    assert ref_out is None and res_out is None


def compute_ref_performance(inputs):
    """Time the reference on independently rebuilt inputs."""
    return measure(compute_ref, lambda: fresh(inputs, kernelgenbench.device))


def compute_res_performance(inputs):
    """Time the candidate on independently rebuilt inputs."""
    return measure(compute_res, lambda: fresh(inputs, kernelgenbench.device))


@label("v1_mask_dp_pad_expert_ids")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_mask_dp_pad_expert_ids(config):
    rows, max_len, top_k, counts, dtype = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    before_counts = None if inputs["counts"] is None else res_inputs["counts"].clone()

    assert compute_ref(**ref_inputs) is None
    assert compute_res(**res_inputs) is None
    compare(None, None)

    expected = _expected(ref_inputs["topk_ids"].clone(), ref_inputs["counts"], max_len)
    assert torch.equal(res_inputs["topk_ids"], expected), "padding mask contract"
    assert torch.equal(res_inputs["topk_ids"], ref_inputs["topk_ids"])
    assert res_inputs["topk_ids"].stride() == inputs["topk_ids"].stride()
    if before_counts is not None:
        assert torch.equal(res_inputs["counts"], before_counts), "counts must not change"

    ref_ms = compute_ref_performance(inputs)
    res_ms = compute_res_performance(inputs)
    return result(ref_ms, res_ms)
