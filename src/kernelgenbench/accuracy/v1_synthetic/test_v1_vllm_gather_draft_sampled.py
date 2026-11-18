"""Accuracy and performance for v1_vllm_gather_draft_sampled.

Contract (vLLM pinned GPU spec-decode): for each logits row, gather the input
token id and position at ``logits_indices``.  A draft slot inside its request's
prefill (``local_pos > 0 and pos - local_pos < prefill_len``) is a placeholder
and its token is replaced by -1 so the rejection kernel rejects it.  Returns a
two-tuple ``(tokens, positions)`` with the dtype of the gathered inputs.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.spec_decode_test_utils import fresh, measure
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (name, positions per token, logits row order, local positions, prefill lengths)
CASES = [
    # Draft slots 1 and 2 of request 0 still sit inside its 2-token prefill.
    ("prefill", [0, 1, 2, 5, 6, 7, 0, 1], [0, 1, 2, 3, 4, 5], [0, 1, 2, 0, 1, 2], [2, 6]),
    # Every row is past its prefill: nothing is a placeholder.
    ("decode", [3, 4, 5, 7, 8, 9, 0, 1], [0, 1, 2, 3, 4, 5], [0, 1, 2, 0, 1, 2], [2, 6]),
    # Reordered logits rows must gather, not assume row == token order.
    ("reordered", [1, 2, 3, 8, 9, 10, 0, 1], [2, 1, 0, 5, 4, 3], [0, 1, 2, 0, 1, 2], [2, 6]),
    # Whole row set is placeholder, including the strict pos - local == prefill boundary.
    ("all-placeholder", [0, 0, 0, 2, 0, 0, 0, 2], [0, 1, 2, 3], [1, 1, 1, 2], [2, 2]),
    # Zero logits rows is a legal no-op shape.
    ("empty", [0, 1, 2, 3, 0, 1, 2, 3], [], [], [2, 2]),
]
COMBOS = [(case, dtype) for case in CASES for dtype in (torch.int32, torch.int64)]
TOKENS = 8
PREFILL = (2, 6)


def input_build(config):
    case, dtype = config
    _, positions, rows, local, prefill = case
    # Token ids are distinct so a wrong gather is visible; positions repeat the
    # request-local position pattern.
    token_ids = torch.arange(10, 10 + TOKENS, dtype=dtype)
    position_values = torch.tensor(positions, dtype=dtype)
    return {
        "input_ids": token_ids,
        "positions": position_values,
        "logits_indices": torch.tensor(rows, dtype=torch.int32),
        "expanded_idx_mapping": torch.tensor([0] * 3 + [1] * 3, dtype=torch.int32)[:len(rows)]
        if len(rows) else torch.empty(0, dtype=torch.int32),
        "expanded_local_pos": torch.tensor(local, dtype=torch.int32),
        "prefill_len": torch.tensor(prefill, dtype=torch.int32),
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_vllm_gather_draft_sampled(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_vllm_gather_draft_sampled(**inputs)


def expected(case, dtype):
    """Independent gather + placeholder mask for this case."""
    _, positions, rows, local, _ = case
    position_values = torch.tensor(positions, dtype=dtype)
    token_ids = torch.arange(10, 10 + TOKENS, dtype=dtype)
    tokens, gathered = [], []
    for row, token_index in enumerate(rows):
        request = 0 if row < 3 else 1
        pos = int(position_values[token_index])
        placeholder = local[row] > 0 and pos - local[row] < PREFILL[request]
        tokens.append(-1 if placeholder else int(token_ids[token_index]))
        gathered.append(pos)
    return tokens, gathered


def compare(ref_out, res_out, dtype):
    assert isinstance(ref_out, tuple) and isinstance(res_out, tuple)
    assert len(ref_out) == len(res_out) == 2
    for name, ref_value, res_value in zip(("tokens", "positions"), ref_out, res_out):
        assert isinstance(res_value, torch.Tensor), name
        assert res_value.dtype == ref_value.dtype == dtype, name
        assert res_value.shape == ref_value.shape, name
        torch.testing.assert_close(res_value, ref_value, rtol=0, atol=0, equal_nan=True)


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs, device=kernelgenbench.device)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs, device=kernelgenbench.device)


@label("v1_vllm_gather_draft_sampled")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_gather_draft_sampled(config):
    case, dtype = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    num_rows = len(case[2])
    for side in (ref_inputs, res_inputs):
        assert side["input_ids"].shape == (TOKENS,)
        assert side["positions"].shape == (TOKENS,)
        assert side["logits_indices"].shape == (num_rows,)
        assert side["expanded_local_pos"].shape == (num_rows,)
        assert side["input_ids"].dtype == dtype

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    tokens, gathered = expected(case, dtype)
    assert ref_out[0].tolist() == tokens, "placeholder masking/gather mismatch"
    assert ref_out[1].tolist() == gathered, "positions gather mismatch"
    # Inputs are read-only; both outputs are fresh tensors.
    for name in ("input_ids", "positions", "logits_indices", "expanded_local_pos"):
        assert torch.equal(ref_inputs[name], inputs[name].to(ref_inputs[name].device))
    assert not torch._C._is_alias_of(ref_out[1], ref_inputs["positions"])

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
