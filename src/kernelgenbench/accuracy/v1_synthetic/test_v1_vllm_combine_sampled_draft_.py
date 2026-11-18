"""Accuracy and performance for v1_vllm_combine_sampled_draft_.

The function writes submitted / draft token IDs into ``input_ids`` in place and
returns the per-logit position indices.  Request states are permuted through
``idx_mapping``; prefill rows keep their prompt tail untouched.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic import rows612_621_test_utils as utils
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

# new sampled token count, cumulative logits per request, sequence lengths
COMBOS = [(1, (0, 3, 5, 7), (5, 9, 11)),
          (0, (0, 3, 5, 7), (8, 9, 12)),
          (1, (0, 1, 2, 3), (7, 8, 10))]


def input_build(config):
    new, cumulative, lengths = config
    tensor = lambda values, dtype: torch.tensor(values, dtype=dtype)
    return {
        "input_ids": tensor(list(range(100, 111)), torch.int32),
        "idx_mapping": tensor([2, 0, 3], torch.int64),
        "last_sampled_tokens": tensor([21, 22, 23, 24], torch.int64),
        "query_start_loc": tensor([0, 4, 8, 11], torch.int32),
        "seq_lens": tensor(lengths, torch.int32),
        "prefill_len": tensor([7, 1, 6, 10], torch.int32),
        "draft_tokens": tensor([[30, 31, 32], [40, 41, 42],
                                [50, 51, 52], [60, 61, 62]], torch.int64),
        "cu_num_logits": tensor(cumulative, torch.int32),
        "num_logits": cumulative[-1],
        "num_new_sampled_tokens": new,
    }


def compute_ref(input_ids, idx_mapping, last_sampled_tokens, query_start_loc,
                seq_lens, prefill_len, draft_tokens, cu_num_logits, num_logits,
                num_new_sampled_tokens):
    return kernelgenbench.baseline.v1_vllm_combine_sampled_draft_(
        input_ids, idx_mapping, last_sampled_tokens, query_start_loc, seq_lens,
        prefill_len, draft_tokens, cu_num_logits, num_logits,
        num_new_sampled_tokens)


def compute_res(input_ids, idx_mapping, last_sampled_tokens, query_start_loc,
                seq_lens, prefill_len, draft_tokens, cu_num_logits, num_logits,
                num_new_sampled_tokens):
    return kernelgenbench.solution.v1_vllm_combine_sampled_draft_(
        input_ids, idx_mapping, last_sampled_tokens, query_start_loc, seq_lens,
        prefill_len, draft_tokens, cu_num_logits, num_logits,
        num_new_sampled_tokens)


def _expected(config):
    new, cumulative, lengths = config
    ids = list(range(100, 111))
    query_start = [0, 4, 8, 11]
    mapping = [2, 0, 3]
    last = [21, 22, 23, 24]
    prefill = [7, 1, 6, 10]
    draft = [[30, 31, 32], [40, 41, 42], [50, 51, 52], [60, 61, 62]]
    out = [0] * cumulative[-1]
    for request, state in enumerate(mapping):
        begin, end = cumulative[request], cumulative[request + 1]
        count = end - begin
        draft_count = count - new
        query_end = query_start[request + 1]
        first = query_end - count
        out[begin:end] = list(range(first, query_end))
        sequence = lengths[request]
        if sequence <= prefill[state]:
            continue
        if new and sequence - count >= prefill[state]:
            ids[first] = last[state]
        if draft_count > 0:
            ids[query_end - draft_count:query_end] = draft[state][:draft_count]
    return out, ids


def compare(ref_out, res_out, config, ref_inputs, res_inputs):
    expected_out, expected_ids = _expected(config)
    for name, out, inputs in (("reference", ref_out, ref_inputs),
                              ("solution", res_out, res_inputs)):
        assert out.shape == (len(expected_out),) and out.dtype == torch.int64, name
        assert out.device == inputs["input_ids"].device, name
        assert out.tolist() == expected_out, (name, out.tolist(), expected_out)
        assert inputs["input_ids"].tolist() == expected_ids, name
    assert torch.equal(ref_out, res_out)
    for key in ("idx_mapping", "last_sampled_tokens", "query_start_loc",
                "seq_lens", "prefill_len", "draft_tokens", "cu_num_logits"):
        assert torch.equal(ref_inputs[key], res_inputs[key]), key


def _timing(config, operation):
    base = input_build(config)
    device = kernelgenbench.device
    return utils.event_time(lambda inputs: operation(**inputs),
                            lambda: utils.fresh(base, device))


def compute_ref_performance(input_ids, idx_mapping, last_sampled_tokens,
                           query_start_loc, seq_lens, prefill_len, draft_tokens,
                           cu_num_logits, num_logits, num_new_sampled_tokens):
    return _timing((num_new_sampled_tokens, tuple(cu_num_logits.tolist()),
                    tuple(seq_lens.tolist())), compute_ref)


def compute_res_performance(input_ids, idx_mapping, last_sampled_tokens,
                           query_start_loc, seq_lens, prefill_len, draft_tokens,
                           cu_num_logits, num_logits, num_new_sampled_tokens):
    return _timing((num_new_sampled_tokens, tuple(cu_num_logits.tolist()),
                    tuple(seq_lens.tolist())), compute_res)


@label("v1_vllm_combine_sampled_draft_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_combine_sampled_draft_(config):
    inputs = input_build(config)
    ref_inputs = utils.fresh(inputs, kernelgenbench.device)
    res_inputs = utils.fresh(inputs, kernelgenbench.device)
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config, ref_inputs, res_inputs)
    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
