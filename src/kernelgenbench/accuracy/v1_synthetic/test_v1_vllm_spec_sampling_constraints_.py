"""Accuracy and performance for v1_vllm_spec_sampling_constraints_.

Temperature is expanded over each request's speculative rows (0 is replaced by
1), then top-k and top-p masks are applied in place.  An all-greedy batch is
returned untouched.  The test checks the in-place identity, the exact mask
counts and the values against the Reference.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic import rows612_621_test_utils as utils
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

# rows per request, vocabulary, temperatures, top_k or None, top_p or None, all greedy
COMBOS = [((2, 0, 3), 9, (0.0, 0.5, 2.0), (4, 9, 3), (0.8, 1.0, 0.5), False),
          ((1, 2), 7, (1.0, 0.5), None, (0.7, 0.4), False),
          ((2, 1), 8, (2.0, 1.0), (2, 6), None, False),
          ((1, 1), 6, (0.5, 2.0), None, None, False),
          ((2, 0), 6, (0.0, 0.0), (2, 2), (0.5, 0.5), True)]


def input_build(config):
    counts, vocab, temperatures, top_k, top_p, greedy = config
    rows = sum(counts)
    logits = ((torch.arange(rows * vocab, dtype=torch.float32).reshape(rows, vocab)
               * 7 % 31) - 15) / 4
    return {
        "logits": logits,
        "cu_num_draft_tokens": torch.tensor(counts, dtype=torch.int64).cumsum(0),
        "temperature": torch.tensor(temperatures, dtype=torch.float32),
        "top_k": None if top_k is None else torch.tensor(top_k, dtype=torch.int32),
        "top_p": None if top_p is None else torch.tensor(top_p, dtype=torch.float32),
        "all_greedy": greedy,
    }


def compute_ref(logits, cu_num_draft_tokens, temperature, top_k, top_p, all_greedy):
    return kernelgenbench.baseline.v1_vllm_spec_sampling_constraints_(
        logits, cu_num_draft_tokens, temperature, top_k, top_p, all_greedy)


def compute_res(logits, cu_num_draft_tokens, temperature, top_k, top_p, all_greedy):
    return kernelgenbench.solution.v1_vllm_spec_sampling_constraints_(
        logits, cu_num_draft_tokens, temperature, top_k, top_p, all_greedy)


def compare(ref_out, res_out, config, ref_inputs, res_inputs):
    counts, vocab, temperatures, top_k, top_p, greedy = config
    assert ref_out is ref_inputs["logits"]
    assert res_out is res_inputs["logits"]
    torch.testing.assert_close(ref_out, res_out, rtol=1e-6, atol=1e-7)
    if greedy:
        torch.testing.assert_close(ref_out, ref_inputs["logits"], rtol=0, atol=0)
        torch.testing.assert_close(res_out, res_inputs["logits"], rtol=0, atol=0)
        return
    begin = 0
    for request, count in enumerate(counts):
        end = begin + count
        if count == 0:
            # A request with no draft rows contributes no logits row at all.
            continue
        row_out = ref_out[begin:end]
        original = ((torch.arange(begin * vocab, end * vocab,
                                  dtype=torch.float32, device=row_out.device)
                     .reshape(-1, vocab) * 7 % 31) - 15) / 4
        temperature = float(temperatures[request]) or 1.0
        scaled = original / temperature
        finite = torch.isfinite(row_out)
        assert bool((torch.isneginf(row_out) | finite).all())
        torch.testing.assert_close(row_out[finite], scaled[finite],
                                   rtol=1e-6, atol=1e-7)
        assert bool((finite.sum(-1) > 0).all())
        if top_k is not None:
            assert int(finite.sum(-1).max()) <= int(top_k[request])
        begin = end


def _timing(config, operation):
    base = input_build(config)
    device = kernelgenbench.device
    return utils.event_time(lambda inputs: operation(**inputs),
                            lambda: utils.fresh(base, device))


def _config_of(logits, cu_num_draft_tokens, temperature, top_k, top_p, all_greedy):
    cumulative = tuple(cu_num_draft_tokens.tolist())
    counts = (cumulative[0],) + tuple(cumulative[i] - cumulative[i - 1]
                                      for i in range(1, len(cumulative)))
    return (counts, logits.shape[1], tuple(temperature.tolist()),
            None if top_k is None else tuple(top_k.tolist()),
            None if top_p is None else tuple(top_p.tolist()), all_greedy)


def compute_ref_performance(logits, cu_num_draft_tokens, temperature, top_k, top_p,
                            all_greedy):
    return _timing(_config_of(logits, cu_num_draft_tokens, temperature, top_k,
                              top_p, all_greedy), compute_ref)


def compute_res_performance(logits, cu_num_draft_tokens, temperature, top_k, top_p,
                            all_greedy):
    return _timing(_config_of(logits, cu_num_draft_tokens, temperature, top_k,
                              top_p, all_greedy), compute_res)


@label("v1_vllm_spec_sampling_constraints_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_spec_sampling_constraints_(config):
    inputs = input_build(config)
    ref_inputs = utils.fresh(inputs, kernelgenbench.device)
    res_inputs = utils.fresh(inputs, kernelgenbench.device)
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config, ref_inputs, res_inputs)
    for key in ("cu_num_draft_tokens", "temperature", "top_k", "top_p"):
        if inputs[key] is not None:
            torch.testing.assert_close(ref_inputs[key].cpu(), inputs[key],
                                       rtol=0, atol=0)
            torch.testing.assert_close(res_inputs[key].cpu(), inputs[key],
                                       rtol=0, atol=0)
    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
