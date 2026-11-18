"""Accuracy and performance for v1_megatron_topk_topp_sampler.

The sampling masses are analytic: temperature scaling, top-k truncation or the
shifted top-p cumulative mask, renormalisation, and the optional vocabulary
clamp.  Greedy rows (top_k == 1) must not consume the controller RNG; random
rows must advance it.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic import rows612_621_test_utils as utils
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

# requests, top_k, top_p, temperature, vocabulary clamp
COMBOS = [(2048, 1, 0.0, 0.5, None), (2048, 3, 0.0, 0.7, None),
          (2048, 0, 0.72, 1.3, None), (2048, 0, 0.0, 1.0, 5)]
BASE = [0.43, 0.23, 0.17, 0.10, 0.05, 0.02]


def input_build(config):
    n, k, p, temperature, vocab = config
    return {
        "probs": torch.tensor(BASE).log().expand(n, -1).clone(),
        "temperature": float(temperature),
        "top_k": int(k),
        "top_p": float(p),
        "generator": None,
        "vocab_size": vocab,
    }


def _generator(inputs, seed=1207):
    device = inputs["probs"].device
    generator = torch.Generator(device=device).manual_seed(seed)
    return generator


def compute_ref(probs, temperature, top_k, top_p, generator, vocab_size):
    return kernelgenbench.baseline.v1_megatron_topk_topp_sampler(
        probs, temperature, top_k, top_p, generator, vocab_size)


def compute_res(probs, temperature, top_k, top_p, generator, vocab_size):
    return kernelgenbench.solution.v1_megatron_topk_topp_sampler(
        probs, temperature, top_k, top_p, generator, vocab_size)


def _mass(config):
    n, k, p, temperature, vocab = config
    if k == 1:
        # Greedy row: the whole mass sits on the argmax token, no RNG consumed.
        one_hot = [0.0] * len(BASE)
        one_hot[max(range(len(BASE)), key=lambda i: BASE[i])] = 1.0
        return one_hot
    mass = [value ** (1.0 / temperature) for value in BASE]
    total = sum(mass)
    mass = [value / total for value in mass]
    if k > 1:
        mass = [value if i < k else 0.0 for i, value in enumerate(mass)]
    elif p > 0:
        cumulative = 0.0
        kept = []
        for value in mass:
            kept.append(value if cumulative <= p else 0.0)
            cumulative += value
        mass = kept
    total = sum(mass)
    mass = [value / total for value in mass]
    if vocab:
        for j in range(vocab, len(mass)):
            mass[vocab - 1] += mass[j]
            mass[j] = 0.0
    return mass


def compare(ref_out, res_out, config):
    n, k, p, temperature, vocab = config
    target = torch.tensor(_mass(config), dtype=torch.float64,
                          device=res_out.device)
    for name, ids in (("reference", ref_out), ("solution", res_out)):
        assert ids.shape == (n,) and ids.dtype == torch.int64, name
        if k == 1:
            assert bool((ids == int(target.argmax())).all()), name
            continue
        counts = torch.bincount(ids, minlength=6).double()
        bound = 8 * torch.sqrt(n * target * (1 - target)) + 4
        assert bool(((counts - n * target).abs() <= bound).all()), (name, counts, target)


def _timing(config, operation):
    base = input_build(config)
    device = kernelgenbench.device

    def prepare():
        inputs = utils.fresh({k: v for k, v in base.items() if k != "generator"},
                             device)
        inputs["generator"] = _generator(inputs)
        return inputs
    return utils.event_time(lambda inputs: operation(**inputs), prepare)


def _config_of(probs, temperature, top_k, top_p, generator, vocab_size):
    return (probs.shape[0], top_k, top_p, temperature, vocab_size)


def compute_ref_performance(probs, temperature, top_k, top_p, generator, vocab_size):
    return _timing(_config_of(probs, temperature, top_k, top_p, generator,
                              vocab_size), compute_ref)


def compute_res_performance(probs, temperature, top_k, top_p, generator, vocab_size):
    return _timing(_config_of(probs, temperature, top_k, top_p, generator,
                              vocab_size), compute_res)


@label("v1_megatron_topk_topp_sampler")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_topk_topp_sampler(config):
    n, k, p, temperature, vocab = config
    inputs = input_build(config)
    ref_inputs = utils.fresh(inputs, kernelgenbench.device)
    res_inputs = utils.fresh(inputs, kernelgenbench.device)
    ref_inputs["generator"] = _generator(ref_inputs)
    res_inputs["generator"] = _generator(res_inputs)
    ref_state = ref_inputs["generator"].get_state().clone()
    res_state = res_inputs["generator"].get_state().clone()
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config)
    if k == 1:
        assert torch.equal(ref_state, ref_inputs["generator"].get_state())
        assert torch.equal(res_state, res_inputs["generator"].get_state())
    else:
        assert not torch.equal(ref_state, ref_inputs["generator"].get_state())
        assert not torch.equal(res_state, res_inputs["generator"].get_state())
    utils.assert_unchanged(inputs, ref_inputs, skip=("generator",))
    utils.assert_unchanged(inputs, res_inputs, skip=("generator",))
    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
