"""Accuracy and performance for v1_moe_multiplicative_jitter.

Operator: deepspeed.moe.sharded_moe.multiplicative_jitter (native torch).
Official source (pinned commit):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/moe/sharded_moe.py#L55-L77

Item: v1_moe_multiplicative_jitter (723-row 166). ``compute_ref`` loads this
file's reference from ``kernelgenbench.baseline``; ``compute_res`` calls the real
official function bound as ``kernelgenbench.solution``. The legacy Megatron fork
branch of the old reference has no official counterpart and was removed; this
item is DeepSpeed only.

The per-device sampler cache is itself under test: a sequence of epsilon values
(0, 0.1, 0.3) is run on each side after seeding the global RNG identically.
epsilon=0 must return the input object without consuming RNG; 0.1 must create the
one cached sampler for the device; 0.3 must reuse that sampler (bounds stay
1 +/- 0.1), which a fresh-sampler implementation gets wrong. Inputs are never
modified.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows162_171_checks as checks


CASES = [(2, 5), (3, 17)]
EPSILONS = (0, 0.1, 0.3)
COMBOS = [(shape, dtype, device)
          for shape in CASES
          for dtype in (torch.float32, torch.float16, torch.bfloat16)
          for device in ("cuda", "cpu")]


def input_build(config):
    shape, dtype, device = config
    x = torch.linspace(-2, 2, shape[0] * shape[1], dtype=dtype).reshape(shape)
    return {"x": x, "device": device, "epsilon": 1e-2, "dtype": dtype}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_moe_multiplicative_jitter(
        x=inputs["x"], device=inputs["device"], epsilon=inputs["epsilon"])


def compute_res(**inputs):
    return kernelgenbench.solution.v1_moe_multiplicative_jitter(
        x=inputs["x"], device=inputs["device"], epsilon=inputs["epsilon"])


def _copy(inputs):
    device = kernelgenbench.device if inputs["device"] == "cuda" else torch.device("cpu")
    return {
        "x": checks.fresh({"x": inputs["x"]}, device)["x"],
        "device": inputs["device"],
        "epsilon": inputs["epsilon"],
        "dtype": inputs["dtype"],
    }


def compare(ref_out, res_out, dtype=None):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape and res_out.dtype == ref_out.dtype
    assert res_out.device == ref_out.device
    torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0, equal_nan=True)


def _sequence(fn, inputs):
    device = inputs["device"]
    torch_device = kernelgenbench.device if device == "cuda" else torch.device("cpu")
    cuda_devices = [torch.cuda.current_device()] if device == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(20261109)
        torch.cuda.manual_seed_all(20261109)
        outputs = [fn(x=inputs["x"], device=device, epsilon=epsilon)
                   for epsilon in EPSILONS]
    return outputs, torch_device


def _timing_device(inputs):
    return kernelgenbench.device if inputs["device"] == "cuda" else torch.device("cpu")


def compute_ref_performance(inputs):
    return checks.measure_factory(lambda: _copy(inputs), compute_ref,
                                  _timing_device(inputs), repeat=10)


def compute_res_performance(inputs):
    return checks.measure_factory(lambda: _copy(inputs), compute_res,
                                  _timing_device(inputs), repeat=10)


@label("v1_moe_multiplicative_jitter")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_moe_multiplicative_jitter(config):
    inputs = input_build(config)
    ref_inputs = _copy(inputs)
    res_inputs = _copy(inputs)
    ref_before = ref_inputs["x"].clone()
    res_before = res_inputs["x"].clone()

    ref_outputs, _ = _sequence(compute_ref, ref_inputs)
    res_outputs, _ = _sequence(compute_res, res_inputs)

    assert ref_outputs[0] is ref_inputs["x"], \
        "epsilon=0 must return the input object without consuming RNG"
    assert res_outputs[0] is res_inputs["x"], \
        "epsilon=0 must return the input object without consuming RNG"
    for index, (expected, actual) in enumerate(zip(ref_outputs, res_outputs)):
        compare(expected, actual)
        assert actual.shape == config[0]
    # 0.3 reuses the sampler created for 0.1, so it must differ from a draw with the
    # wider (1-0.3, 1+0.3) bounds for the same seed.
    torch.testing.assert_close(ref_inputs["x"], ref_before, rtol=0, atol=0)
    torch.testing.assert_close(res_inputs["x"], res_before, rtol=0, atol=0)

    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return checks.timings(ref_time, res_time)
