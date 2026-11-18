"""Accuracy and performance for v1_sgl_topk_topp_minp_sample.

Every request repeats the same six-way distribution so the sampled histogram can
be checked against the filtered distribution.  The assertions cover the exact
filtered probability matrix/permutation, the sampled-ID/permutation relation,
the selected weight, and the categorical histogram.  The two sides draw
independent random samples, so values are never compared across sides.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic import rows612_621_test_utils as utils
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

# requests, top_k, top_p, min_p, enable_min_p_sampling
COMBOS = [(2048, 6, 1.0, 0.0, False),
          (2048, 3, 0.70, 0.0, False),
          (2048, 4, 0.90, 0.4, True)]
PROBS = [0.43, 0.23, 0.17, 0.10, 0.05, 0.02]


def input_build(config):
    n, k, p, min_p, enabled = config
    return {
        "probs": torch.tensor(PROBS).expand(n, -1).clone(),
        "top_ks": torch.full((n,), k, dtype=torch.int32),
        "top_ps": torch.full((n,), p),
        "min_ps": torch.full((n,), min_p),
        "need_min_p_sampling": enabled,
    }


def compute_ref(probs, top_ks, top_ps, min_ps, need_min_p_sampling):
    return kernelgenbench.baseline.v1_sgl_topk_topp_minp_sample(
        probs, top_ks, top_ps, min_ps, need_min_p_sampling)


def compute_res(probs, top_ks, top_ps, min_ps, need_min_p_sampling):
    return kernelgenbench.solution.v1_sgl_topk_topp_minp_sample(
        probs, top_ks, top_ps, min_ps, need_min_p_sampling)


def compare(ref_out, res_out, config):
    n = config[0]
    assert len(ref_out) == len(res_out) == 4
    # The filtered matrix and its descending-token permutation are deterministic.
    torch.testing.assert_close(ref_out[1], res_out[1], rtol=0, atol=0)
    assert torch.equal(ref_out[2], res_out[2])
    for name, out in (("reference", ref_out), ("solution", res_out)):
        ids, weights, perm, selected = out
        assert ids.shape == (n,) and ids.dtype == torch.int32, name
        assert weights.shape == (n, 6) and weights.dtype == torch.float32, name
        assert perm.shape == (n, 6) and perm.dtype == torch.int32, name
        assert selected.shape == (n,), name
        where = perm == ids[:, None]
        assert bool((where.sum(1) == 1).all()), name
        assert bool((weights[where] > 0).all()), name
        torch.testing.assert_close(selected, weights[where])
        probability = weights[0].double() / weights[0].double().sum()
        counts = torch.bincount(ids.long(), minlength=6).double()
        target = torch.zeros(6, dtype=torch.float64, device=ids.device)
        target[perm[0].long()] = probability
        bound = 8 * torch.sqrt(n * target * (1 - target)) + 4
        assert bool(((counts - n * target).abs() <= bound).all()), (name, counts, target)


def _timing(config, operation):
    base = input_build(config)
    device = kernelgenbench.device
    return utils.event_time(lambda inputs: operation(**inputs),
                            lambda: utils.fresh(base, device))


def compute_ref_performance(probs, top_ks, top_ps, min_ps, need_min_p_sampling):
    return _timing((probs.shape[0], int(top_ks[0]), float(top_ps[0]),
                    float(min_ps[0]), need_min_p_sampling), compute_ref)


def compute_res_performance(probs, top_ks, top_ps, min_ps, need_min_p_sampling):
    return _timing((probs.shape[0], int(top_ks[0]), float(top_ps[0]),
                    float(min_ps[0]), need_min_p_sampling), compute_res)


@label("v1_sgl_topk_topp_minp_sample")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_topk_topp_minp_sample(config):
    inputs = input_build(config)
    ref_inputs = utils.fresh(inputs, kernelgenbench.device)
    res_inputs = utils.fresh(inputs, kernelgenbench.device)
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config)
    # Both implementations filter in local sorted copies; the inputs stay intact.
    utils.assert_unchanged(inputs, ref_inputs)
    utils.assert_unchanged(inputs, res_inputs)
    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
