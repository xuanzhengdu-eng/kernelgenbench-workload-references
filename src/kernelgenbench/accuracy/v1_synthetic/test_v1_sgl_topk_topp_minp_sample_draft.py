"""Accuracy and performance for v1_sgl_topk_topp_minp_sample_draft.

q is deterministic (softmax of logits / temperature); the drawn index is either
a categorical sample from q or, for top_k <= 1 rows, the argmax of q.  The
assertions check q, q(X), the greedy override and the sampled histogram.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic import rows612_621_test_utils as utils
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

# requests, temperature, top_k pattern (none / mixed / all-greedy)
COMBOS = [(2048, 1.0, "none"), (2048, 0.8, "mixed"), (2048, 1.5, "all")]
LOGITS = [1.5, 0.8, 0.4, 0.1, -0.3, -0.8]


def input_build(config):
    n, temperature, pattern = config
    top_ks = None if pattern == "none" else torch.full(
        (n,), 1 if pattern == "all" else 6, dtype=torch.int32)
    if pattern == "mixed":
        top_ks[::2] = 1
    return {
        "next_token_logits": torch.tensor(LOGITS).expand(n, -1).clone(),
        "temperatures": torch.full((n, 1), temperature),
        "top_ks": top_ks,
    }


def compute_ref(next_token_logits, temperatures, top_ks):
    return kernelgenbench.baseline.v1_sgl_topk_topp_minp_sample_draft(
        next_token_logits, temperatures, top_ks)


def compute_res(next_token_logits, temperatures, top_ks):
    return kernelgenbench.solution.v1_sgl_topk_topp_minp_sample_draft(
        next_token_logits, temperatures, top_ks)


def compare(ref_out, res_out, config):
    n, temperature, pattern = config
    torch.testing.assert_close(ref_out[0], res_out[0], rtol=0, atol=0)
    for name, out in (("reference", ref_out), ("solution", res_out)):
        q, selected, index = out
        assert q.shape == (n, 6) and q.dtype == torch.float32, name
        assert selected.shape == (n, 1) and index.shape == (n, 1), name
        assert index.dtype == torch.int64, name
        torch.testing.assert_close(selected, q.gather(1, index), rtol=0, atol=0)
        greedy = torch.zeros(n, dtype=torch.bool) if pattern == "none" else (
            torch.ones(n, dtype=torch.bool) if pattern == "all" else
            (torch.arange(n) % 2 == 0))
        assert torch.equal(index[greedy], q[greedy].argmax(-1, keepdim=True)), name
        drawn = index[~greedy].flatten()
        if drawn.numel():
            counts = torch.bincount(drawn, minlength=6).double()
            target = q[0].double()
            bound = 8 * torch.sqrt(drawn.numel() * target * (1 - target)) + 4
            assert bool(((counts - drawn.numel() * target).abs() <= bound).all()), \
                (name, counts, target)


def _timing(config, operation):
    base = input_build(config)
    device = kernelgenbench.device
    return utils.event_time(lambda inputs: operation(**inputs),
                            lambda: utils.fresh(base, device))


def compute_ref_performance(next_token_logits, temperatures, top_ks):
    return _timing((next_token_logits.shape[0], float(temperatures[0, 0]),
                    "none" if top_ks is None else
                    ("all" if bool((top_ks == 1).all()) else "mixed")), compute_ref)


def compute_res_performance(next_token_logits, temperatures, top_ks):
    return _timing((next_token_logits.shape[0], float(temperatures[0, 0]),
                    "none" if top_ks is None else
                    ("all" if bool((top_ks == 1).all()) else "mixed")), compute_res)


@label("v1_sgl_topk_topp_minp_sample_draft")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_topk_topp_minp_sample_draft(config):
    inputs = input_build(config)
    ref_inputs = utils.fresh(inputs, kernelgenbench.device)
    res_inputs = utils.fresh(inputs, kernelgenbench.device)
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config)
    utils.assert_unchanged(inputs, ref_inputs)
    utils.assert_unchanged(inputs, res_inputs)
    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
