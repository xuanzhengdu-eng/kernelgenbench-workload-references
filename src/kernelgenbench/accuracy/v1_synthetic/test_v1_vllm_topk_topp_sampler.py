"""Accuracy and performance for v1_vllm_topk_topp_sampler.

The native path filters the logits in place, so the test observes the processed
matrix (top-k threshold plus shifted top-p cumulative mask), the aliasing of the
processed-logits return value and the log-softmax return mode.  Sampled IDs are
checked against the analytic categorical distribution; two rows use explicit
per-request generators.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic import rows612_621_test_utils as utils
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

# requests, top_k or None, top_p or None, return mode, per-request generators, fp64 noise
COMBOS = [(2048, 3, None, "processed_logits", False, False),
          (2048, None, 0.7, "processed_logprobs", False, False),
          (2048, 4, 0.85, "raw_logprobs", True, True),
          (2048, None, None, "processed_logits", False, False)]
BASE = [0.43, 0.23, 0.17, 0.10, 0.05, 0.02]


def input_build(config):
    n, k, p, mode, seeded, fp64 = config
    # Seeds only: each side builds its own equal generator on the logits device.
    generators = {i: 330 + i for i in range(2)} if seeded else {}
    return {
        "logits": torch.tensor(BASE).log().expand(n, -1).clone(),
        "top_ks": None if k is None else torch.full((n,), k, dtype=torch.int32),
        "top_ps": None if p is None else torch.full((n,), p),
        "generators": generators,
        "logprobs_mode": mode,
        "use_fp64_gumbel": fp64,
    }


def _generators(generators, device):
    return {row: torch.Generator(device=device).manual_seed(seed)
            for row, seed in generators.items()}


def compute_ref(logits, top_ks, top_ps, generators, logprobs_mode, use_fp64_gumbel):
    return kernelgenbench.baseline.v1_vllm_topk_topp_sampler(
        logits, top_ks, top_ps, _generators(generators, logits.device),
        logprobs_mode, use_fp64_gumbel)


def compute_res(logits, top_ks, top_ps, generators, logprobs_mode, use_fp64_gumbel):
    return kernelgenbench.solution.v1_vllm_topk_topp_sampler(
        logits, top_ks, top_ps, _generators(generators, logits.device),
        logprobs_mode, use_fp64_gumbel)


def _filtered(config, device):
    n, k, p, mode, seeded, fp64 = config
    mass = list(BASE)
    if k is not None:
        mass = [value if i < k else 0.0 for i, value in enumerate(mass)]
    if p is not None:
        total = sum(mass)
        cumulative = 0.0
        for j in reversed(range(len(mass))):
            cumulative += mass[j] / total
            if j != 0 and cumulative <= 1.0 - p:
                mass[j] = 0.0
    total = sum(mass)
    mass = [value / total for value in mass]
    # input_build already stores logits; reuse that exact float32 buffer.
    expected = input_build(config)["logits"][0].clone()
    expected[torch.tensor(mass) == 0] = float("-inf")
    return expected.to(device).expand(n, -1), mass


def compare(ref_out, res_out, config, ref_inputs, res_inputs):
    n, k, p, mode, seeded, fp64 = config
    assert len(ref_out) == len(res_out) == 2
    expected, mass = _filtered(config, ref_inputs["logits"].device)
    # Both sides apply the same in-place mask to their own logits buffer.
    torch.testing.assert_close(ref_inputs["logits"], expected, rtol=0, atol=0)
    torch.testing.assert_close(res_inputs["logits"], expected, rtol=0, atol=0)
    for name, out, inputs in (("reference", ref_out, ref_inputs),
                              ("solution", res_out, res_inputs)):
        ids, extra = out
        assert ids.shape == (n,) and ids.dtype == torch.int64, name
        if mode == "processed_logits":
            assert extra is inputs["logits"], name
        elif mode == "processed_logprobs":
            torch.testing.assert_close(
                extra, expected.log_softmax(-1, dtype=torch.float32), rtol=0, atol=0)
        else:
            assert extra is None, name
        counts = torch.bincount(ids, minlength=6).double()
        target = torch.tensor(mass, dtype=torch.float64, device=ids.device)
        bound = 8 * torch.sqrt(n * target * (1 - target)) + 4
        assert bool(((counts - n * target).abs() <= bound).all()), (name, counts, target)


def _timing(config, operation):
    base = input_build(config)
    device = kernelgenbench.device
    return utils.event_time(lambda inputs: operation(**inputs),
                            lambda: utils.fresh(base, device))


def _config_of(logits, top_ks, top_ps, generators, logprobs_mode, use_fp64_gumbel):
    return (logits.shape[0], None if top_ks is None else int(top_ks[0]),
            None if top_ps is None else float(top_ps[0]), logprobs_mode,
            bool(generators), use_fp64_gumbel)


def compute_ref_performance(logits, top_ks, top_ps, generators, logprobs_mode,
                            use_fp64_gumbel):
    return _timing(_config_of(logits, top_ks, top_ps, generators, logprobs_mode,
                              use_fp64_gumbel), compute_ref)


def compute_res_performance(logits, top_ks, top_ps, generators, logprobs_mode,
                            use_fp64_gumbel):
    return _timing(_config_of(logits, top_ks, top_ps, generators, logprobs_mode,
                              use_fp64_gumbel), compute_res)


@label("v1_vllm_topk_topp_sampler")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_topk_topp_sampler(config):
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
