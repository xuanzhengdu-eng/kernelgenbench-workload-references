"""Accuracy and timing for v1_mask_empty_dcp_lse_.

Each sequence owns ``query_start_loc[i] .. query_start_loc[i + 1]`` rows.
Rows belonging to a zero-length sequence, or trailing padded rows, become
negative infinity; everything else is untouched.  The function mutates the
LSE tensor in place and returns None, and must not touch either metadata
tensor.  A paired None disables masking.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows642_651_test_utils import (
    clone_tree,
    measure_ms,
    move_tree,
)
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


DTYPES = (torch.float32, torch.float16, torch.bfloat16)

# (lse rows, seq_lens, query_start_loc, disabled, dtype)
COMBOS = [
    (rows, lengths, starts, False, dtype)
    for rows, lengths, starts in [
        (8, (4, 0, 2), (0, 2, 4, 5)),
        (5, (), (0,)),
        (6, (1, 2, 0), (0, 0, 3, 3)),
        (3, (2,), (0, 3)),
        (4, (0, 0), (0, 0, 2)),
        (0, (1,), (0, 1)),
    ]
    for dtype in DTYPES
] + [
    (8, (4, 0, 2), (0, 2, 4, 5), True, torch.float32),
    (5, (), (0,), True, torch.float16),
]


def input_build(config):
    rows, lengths, starts, disabled, dtype = config
    return {
        "lse": torch.arange(rows * 4, dtype=dtype).reshape(rows, 4),
        "seq_lens": (
            None
            if disabled
            else torch.tensor(lengths, dtype=torch.int32)
        ),
        "query_start_loc": (
            None
            if disabled
            else torch.tensor(starts, dtype=torch.int32)
        ),
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_mask_empty_dcp_lse_(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_mask_empty_dcp_lse_(**inputs)


def _expected(lse, seq_lens, query_start_loc):
    if seq_lens is None:
        return lse.clone()
    if seq_lens.numel() == 0:
        return torch.full_like(lse, float("-inf"))
    expected = lse.clone()
    starts = query_start_loc.tolist()
    lengths = seq_lens.tolist()
    for row in range(lse.shape[0]):
        # A row at a repeated boundary belongs to the next sequence.
        sequence = 0
        while sequence + 1 < len(starts) and row >= starts[sequence + 1]:
            sequence += 1
        sequence = min(sequence, len(lengths) - 1)
        if row >= starts[-1] or lengths[sequence] == 0:
            expected[row].fill_(float("-inf"))
    return expected


def compare(ref_out, res_out, source_lse, ref_inputs, res_inputs):
    assert ref_out is None and res_out is None
    expected = _expected(
        source_lse,
        ref_inputs["seq_lens"],
        ref_inputs["query_start_loc"],
    )
    assert res_inputs["lse"].dtype == ref_inputs["lse"].dtype
    assert res_inputs["lse"].shape == ref_inputs["lse"].shape
    torch.testing.assert_close(
        res_inputs["lse"], expected, rtol=0, atol=0
    )
    torch.testing.assert_close(
        ref_inputs["lse"], expected, rtol=0, atol=0
    )


def compute_ref_performance(**inputs):
    return measure_ms(compute_ref, inputs, kernelgenbench.device)


def compute_res_performance(**inputs):
    return measure_ms(compute_res, inputs, kernelgenbench.device)


@label("v1_mask_empty_dcp_lse_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_mask_empty_dcp_lse_(config):
    inputs = input_build(config)
    ref_inputs = move_tree(clone_tree(inputs), kernelgenbench.device)
    res_inputs = move_tree(clone_tree(inputs), kernelgenbench.device)
    ref_before = clone_tree(ref_inputs)
    res_before = clone_tree(res_inputs)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(
        ref_out, res_out, ref_before["lse"], ref_inputs, res_inputs
    )

    for name in ("seq_lens", "query_start_loc"):
        if ref_inputs[name] is not None:
            torch.testing.assert_close(
                ref_inputs[name].cpu(), ref_before[name].cpu(), rtol=0, atol=0
            )
            torch.testing.assert_close(
                res_inputs[name].cpu(), res_before[name].cpu(), rtol=0, atol=0
            )

    ref_time = compute_ref_performance(**ref_inputs)
    res_time = compute_res_performance(**res_inputs)
    return CustomBenchmarkResult(
        ref_time=ref_time,
        res_time=res_time,
        speedup=ref_time / res_time if res_time else float("inf"),
    )


def test_mask_empty_dcp_lse_requires_paired_metadata():
    lse = torch.zeros((2, 4))
    seq_lens = torch.tensor([2], dtype=torch.int32)
    with pytest.raises(ValueError):
        compute_ref(lse=lse, seq_lens=seq_lens, query_start_loc=None)
    with pytest.raises(ValueError):
        compute_res(lse=lse, seq_lens=seq_lens, query_start_loc=None)


def test_mask_empty_dcp_lse_rejects_short_query_start_loc():
    lse = torch.zeros((2, 4))
    seq_lens = torch.tensor([2, 0], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2], dtype=torch.int32)
    with pytest.raises(ValueError):
        compute_ref(lse=lse, seq_lens=seq_lens, query_start_loc=query_start_loc)
    with pytest.raises(ValueError):
        compute_res(lse=lse, seq_lens=seq_lens, query_start_loc=query_start_loc)
