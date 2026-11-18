"""Accuracy and performance for v1_sgl_topk_topp_minp_sample_seeded_draft.

The seeded path is deterministic: the same (seed, position, draft step) must
produce the same hash, the same q and the same drafted index.  Rows with
top_k <= 1 are forced to the argmax of q.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic import rows612_621_test_utils as utils
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

# requests, vocabulary, draft step, force every other row greedy
COMBOS = [(8, 6, 0, False), (7, 17, 1, True), (6, 32, 5, False)]


def input_build(config):
    n, v, step, mixed = config
    generator = torch.Generator().manual_seed(1234 + v + step)
    logits = torch.randn(n, v, generator=generator)
    top_ks = torch.full((n,), v, dtype=torch.int32)
    if mixed:
        top_ks[::2] = 1
    return {
        "probs": logits,
        "temperatures": torch.linspace(0.5, 1.5, n).view(-1, 1),
        "top_ks": top_ks,
        "sampling_seed": torch.arange(n, dtype=torch.int64) * 1009 + 2 ** 40,
        "positions": torch.arange(n, dtype=torch.int64) * 17 + 5,
        "draft_step": step,
    }


def compute_ref(probs, temperatures, top_ks, sampling_seed, positions, draft_step):
    return kernelgenbench.baseline.v1_sgl_topk_topp_minp_sample_seeded_draft(
        probs, temperatures, top_ks, sampling_seed, positions, draft_step)


def compute_res(probs, temperatures, top_ks, sampling_seed, positions, draft_step):
    return kernelgenbench.solution.v1_sgl_topk_topp_minp_sample_seeded_draft(
        probs, temperatures, top_ks, sampling_seed, positions, draft_step)


def compare(ref_out, res_out, config):
    n, v, step, mixed = config
    for name, out in (("reference", ref_out), ("solution", res_out)):
        q, selected, index = out
        assert q.shape == (n, v) and q.dtype == torch.float32, name
        assert selected.shape == (n, 1) and index.shape == (n, 1), name
        assert index.dtype == torch.int64, name
        torch.testing.assert_close(selected, q.gather(1, index), rtol=0, atol=0)
        if mixed:
            greedy = torch.zeros(n, dtype=torch.bool)
            greedy[::2] = True
            assert torch.equal(index[greedy],
                               q[greedy].argmax(-1, keepdim=True)), name
    torch.testing.assert_close(ref_out[0], res_out[0], rtol=0, atol=0)
    assert torch.equal(ref_out[1], res_out[1])
    assert torch.equal(ref_out[2], res_out[2])


def _timing(config, operation):
    base = input_build(config)
    device = kernelgenbench.device
    return utils.event_time(lambda inputs: operation(**inputs),
                            lambda: utils.fresh(base, device),
                            warmup=5, repetitions=20)


def compute_ref_performance(probs, temperatures, top_ks, sampling_seed, positions,
                            draft_step):
    return _timing((probs.shape[0], probs.shape[1], draft_step, bool((top_ks == 1).any())),
                   compute_ref)


def compute_res_performance(probs, temperatures, top_ks, sampling_seed, positions,
                            draft_step):
    return _timing((probs.shape[0], probs.shape[1], draft_step, bool((top_ks == 1).any())),
                   compute_res)


@label("v1_sgl_topk_topp_minp_sample_seeded_draft")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_topk_topp_minp_sample_seeded_draft(config):
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
