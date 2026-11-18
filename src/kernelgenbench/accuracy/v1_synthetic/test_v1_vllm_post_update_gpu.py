"""Accuracy and performance for the GPU/Triton branch of vLLM sampling post-update.

Pinned entry ``vllm.v1.worker.gpu.input_batch.post_update``.  Same request-state
bookkeeping as the CPU C++ operator, but executed by the Triton
``_post_update_kernel`` with one program per request row.  GPU contract: CUDA
tensors, unique non-negative request state ids (duplicates race across
programs), int64 ``last_sampled_tokens`` and int64 ``sampled_tokens``; the
histogram, token table and total length may be strided.  Returns None.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.spec_decode_test_utils import fresh, measure
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (histogram, query metadata, strided state rows, extra request row)
CASES = [
    (True, True, False, False),
    (False, False, True, False),
    (True, True, True, True),
    (False, True, False, False),
]
COMBOS = CASES
FIELDS = ("num_computed_tokens", "last_sampled_tokens", "output_bin_counts",
          "all_token_ids", "total_len")


def input_build(config):
    counts, query, strided, extra = config
    batch = 4 if extra else 3
    # Unique state ids per live row: 1, -1, 0[, 2].
    idx = torch.tensor([1, -1, 0, 2][:batch], dtype=torch.int64)
    computed = torch.tensor([4, 5, 6, 7][:batch], dtype=torch.int32)
    last = torch.tensor([-9, -8, -7, -6][:batch], dtype=torch.int64)
    hist = torch.zeros((batch, 8), dtype=torch.int32) if counts else None
    samples = torch.tensor([[2, 2, 0], [3, 4, 0], [4, 0, 0], [5, 6, 7]][:batch],
                           dtype=torch.int64)
    all_token_ids = torch.full((batch, 8), -1, dtype=torch.int32)
    if strided:
        # Wider backing rows for the histogram and token table (stride honoured).
        if hist is not None:
            wider = torch.zeros((batch, 10), dtype=torch.int32)
            hist = wider[:, :8]
        wider_ids = torch.full((batch, 10), -1, dtype=torch.int32)
        all_token_ids = wider_ids[:, :8]
    return {
        "idx_mapping": idx,
        "num_computed_tokens": computed,
        "last_sampled_tokens": last,
        "output_bin_counts": hist,
        "sampled_tokens": samples,
        "num_sampled": torch.tensor([2, 1, 0, 2][:batch], dtype=torch.int32),
        "num_rejected": torch.tensor([1, 0, 2, 0][:batch], dtype=torch.int32),
        "query_start_loc": torch.tensor([0, 3, 4, 6, 8][:batch + 1], dtype=torch.int32)
        if query else None,
        "all_token_ids": all_token_ids,
        "total_len": torch.tensor([2, 1, 3, 4][:batch], dtype=torch.int32),
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_vllm_post_update_gpu(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_vllm_post_update_gpu(**inputs)


def compare(ref_out, res_out, dtype=None):
    assert ref_out is None and res_out is None


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs, device=kernelgenbench.device)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs, device=kernelgenbench.device)


@label("v1_vllm_post_update_gpu")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_post_update_gpu(config):
    counts, query, strided, extra = config
    batch = 4 if extra else 3
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    before = {name: value.clone() for name, value in inputs.items()
              if isinstance(value, torch.Tensor)}
    assert ref_inputs["idx_mapping"].device.type == "cuda"
    assert ref_inputs["last_sampled_tokens"].is_contiguous()
    assert ref_inputs["last_sampled_tokens"].shape == (batch,)

    ref_live = {name: ref_inputs[name] for name in FIELDS}
    res_live = {name: res_inputs[name] for name in FIELDS}
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out)
    for name in FIELDS:
        assert ref_inputs[name] is ref_live[name], f"{name} was rebound on the reference side"
        assert res_inputs[name] is res_live[name], f"{name} was rebound on the candidate side"

    for name in FIELDS:
        expected, actual = ref_inputs[name], res_inputs[name]
        if expected is None:
            assert actual is None, name
            continue
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, name
        assert torch.equal(actual, expected), f"{name} differs from reference"

    # Sampled tokens land at each request's old total length and bump the
    # histogram, last sampled token, total length and computed-token delta.
    for row, req in enumerate(inputs["idx_mapping"].tolist()):
        if req < 0:
            continue
        count = int(inputs["num_sampled"][row])
        old_total = int(before["total_len"][req])
        if count:
            assert int(res_inputs["last_sampled_tokens"][req]) == int(
                inputs["sampled_tokens"][row, count - 1])
            assert int(res_inputs["total_len"][req]) == old_total + count
        tally = {}
        for offset in range(count):
            assert int(res_inputs["all_token_ids"][req, old_total + offset]) == int(
                inputs["sampled_tokens"][row, offset])
            if counts:
                token = int(inputs["sampled_tokens"][row, offset])
                tally[token] = tally.get(token, 0) + 1
        for token, added in tally.items():
            assert int(res_inputs["output_bin_counts"][req, token]) == int(
                before["output_bin_counts"][req, token]) + added
        query_len = 0 if not query else (
            int(inputs["query_start_loc"][row + 1]) - int(inputs["query_start_loc"][row]))
        assert int(res_inputs["num_computed_tokens"][req]) == (
            int(before["num_computed_tokens"][req]) + query_len
            - int(inputs["num_rejected"][row]))

    if strided:
        if counts:
            assert res_inputs["output_bin_counts"].stride(0) == 10
        assert res_inputs["all_token_ids"].stride(0) == 10

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
