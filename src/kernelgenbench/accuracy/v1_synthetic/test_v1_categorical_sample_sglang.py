"""Accuracy and performance for v1_categorical_sample_sglang.

sampling_from_probs_torch without a seed samples with torch.multinomial (or its
gumbel-max equivalent) and returns int32 token IDs.  The probability input is
read-only; the histogram must match the shared six-way distribution.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic import rows612_621_test_utils as utils
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

# requests; every row repeats the same six-way distribution.
COMBOS = [(2048,)]
PROBS = [0.43, 0.23, 0.17, 0.10, 0.05, 0.02]


def input_build(config):
    n = config[0]
    return {"probs": torch.tensor(PROBS).expand(n, -1).clone()}


def compute_ref(probs):
    return kernelgenbench.baseline.v1_categorical_sample_sglang(probs)


def compute_res(probs):
    return kernelgenbench.solution.v1_categorical_sample_sglang(probs)


def compare(ref_out, res_out, config):
    n = config[0]
    target = torch.tensor(PROBS, dtype=torch.float64, device=ref_out.device) / sum(PROBS)
    for name, ids in (("reference", ref_out), ("solution", res_out)):
        assert ids.shape == (n,) and ids.dtype == torch.int32, name
        assert bool(((ids >= 0) & (ids < 6)).all()), name
        counts = torch.bincount(ids.long(), minlength=6).double()
        bound = 8 * torch.sqrt(n * target * (1 - target)) + 4
        assert bool(((counts - n * target).abs() <= bound).all()), (name, counts, target)


def _timing(config, operation):
    base = input_build(config)
    device = kernelgenbench.device
    return utils.event_time(lambda inputs: operation(**inputs),
                            lambda: utils.fresh(base, device))


def compute_ref_performance(probs):
    return _timing((probs.shape[0],), compute_ref)


def compute_res_performance(probs):
    return _timing((probs.shape[0],), compute_res)


@label("v1_categorical_sample_sglang")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_categorical_sample_sglang(config):
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
