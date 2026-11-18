"""Accuracy and performance for v1_categorical_sample_vllm.

random_sample draws exponential noise and argmaxes probs / noise.  With float32
noise the scores are written back into probs (the returned IDs equal
probs.argmax(-1) afterwards); with float64 noise the probability input must stay
untouched.  Both variants must match the shared six-way histogram.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic import rows612_621_test_utils as utils
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

# requests, use float64 exponential noise
COMBOS = [(2048, False), (2048, True)]
PROBS = [0.43, 0.23, 0.17, 0.10, 0.05, 0.02]


def input_build(config):
    n, fp64 = config
    return {"probs": torch.tensor(PROBS).expand(n, -1).clone(),
            "use_fp64_gumbel": fp64}


def compute_ref(probs, use_fp64_gumbel):
    return kernelgenbench.baseline.v1_categorical_sample_vllm(probs, use_fp64_gumbel)


def compute_res(probs, use_fp64_gumbel):
    return kernelgenbench.solution.v1_categorical_sample_vllm(probs, use_fp64_gumbel)


def compare(ref_out, res_out, config, ref_inputs, res_inputs):
    n, fp64 = config
    original = torch.tensor(PROBS, dtype=torch.float64, device=ref_out.device)
    original = original / original.sum()
    for name, ids, inputs in (("reference", ref_out, ref_inputs),
                              ("solution", res_out, res_inputs)):
        assert ids.shape == (n,) and ids.dtype == torch.int64, name
        assert bool(((ids >= 0) & (ids < 6)).all()), name
        if fp64:
            torch.testing.assert_close(inputs["probs"],
                                       torch.tensor(PROBS,
                                                    device=inputs["probs"].device)
                                       .expand(n, -1), rtol=0, atol=0)
        else:
            assert bool((inputs["probs"] >= 0).all()), name
            assert bool((inputs["probs"].sum(-1) > 0).all()), name
            assert torch.equal(ids, inputs["probs"].argmax(-1)), name
        counts = torch.bincount(ids, minlength=6).double()
        bound = 8 * torch.sqrt(n * original * (1 - original)) + 4
        assert bool(((counts - n * original).abs() <= bound).all()), (name, counts)


def _timing(config, operation):
    base = input_build(config)
    device = kernelgenbench.device
    return utils.event_time(lambda inputs: operation(**inputs),
                            lambda: utils.fresh(base, device))


def compute_ref_performance(probs, use_fp64_gumbel):
    return _timing((probs.shape[0], use_fp64_gumbel), compute_ref)


def compute_res_performance(probs, use_fp64_gumbel):
    return _timing((probs.shape[0], use_fp64_gumbel), compute_res)


@label("v1_categorical_sample_vllm")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_categorical_sample_vllm(config):
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
