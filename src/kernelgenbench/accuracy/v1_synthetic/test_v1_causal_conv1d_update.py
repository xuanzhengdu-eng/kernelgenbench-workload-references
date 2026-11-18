"""Accuracy and CPU performance for v1_causal_conv1d_update.

The input layout, in-place cache update and optional SiLU are visible below.
The reference is the pinned vLLM CPU PyTorch fallback; the solution is bound by
the verifier to the same official entry and must update an independent cache.
"""
import time

import pytest
import torch
import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

CASES = [(2, 64, 1, 2, 1), (2, 64, 3, 4, 4)]
DTYPES = (torch.float32, torch.bfloat16)
COMBOS = [
    (case, dtype, bias_present, activation)
    for case in CASES
    for dtype in DTYPES
    for bias_present in (False, True)
    for activation in (None, "silu")
]


def input_build(config):
    (batch, channels, seq, width, state_len), dtype, bias_present, activation = config
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    return {
        "x": torch.randn((batch, channels, seq), dtype=dtype, generator=generator),
        "conv_state": torch.randn((batch, channels, state_len), dtype=dtype, generator=generator),
        "weight": torch.randn((channels, width), dtype=dtype, generator=generator),
        "bias": (torch.randn((channels,), dtype=dtype, generator=generator)
                 if bias_present else None),
        "activation": activation,
    }


def fresh(inputs):
    return {name: value.clone() if isinstance(value, torch.Tensor) else value
            for name, value in inputs.items()}


def compute_ref(x, conv_state, weight, bias, activation):
    return kernelgenbench.baseline.v1_causal_conv1d_update(
        x, conv_state, weight, bias, activation)


def compute_res(x, conv_state, weight, bias, activation):
    return kernelgenbench.solution.v1_causal_conv1d_update(
        x, conv_state, weight, bias, activation)


def compare(ref_out, res_out, ref_state, res_state, dtype):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == dtype
    torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0)
    # The cache is the stateful part of the contract: both sides must write the
    # same shifted window, not merely return the same output.
    assert torch.equal(res_state, ref_state)


def time_cpu(fn, inputs, iters=300):
    state = inputs["conv_state"].clone()
    for _ in range(20):
        inputs["conv_state"].copy_(state)
        fn(**inputs)
    elapsed = 0.0
    for _ in range(iters):
        inputs["conv_state"].copy_(state)
        start = time.perf_counter()
        fn(**inputs)
        elapsed += time.perf_counter() - start
    return elapsed / iters


def compute_ref_performance(x, conv_state, weight, bias, activation):
    return time_cpu(compute_ref, {"x": x, "conv_state": conv_state, "weight": weight,
                                  "bias": bias, "activation": activation})


def compute_res_performance(x, conv_state, weight, bias, activation):
    return time_cpu(compute_res, {"x": x, "conv_state": conv_state, "weight": weight,
                                  "bias": bias, "activation": activation})


@label("v1_causal_conv1d_update")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_causal_conv1d_update(config):
    case, dtype, _, _ = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs)
    res_inputs = fresh(inputs)

    batch, channels, seq, _, _ = case
    assert ref_inputs["x"].shape == (batch, channels, seq)
    assert ref_inputs["conv_state"].shape[-1] == case[-1]
    assert ref_inputs["weight"].shape == (channels, case[3])

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, ref_inputs["conv_state"], res_inputs["conv_state"], dtype)

    assert ref_out.shape == (batch, channels, seq)
    assert not torch._C._is_alias_of(ref_out, ref_inputs["x"])
    assert not torch._C._is_alias_of(res_out, res_inputs["x"])
    for name in ("x", "weight", "bias"):
        if inputs[name] is not None:
            torch.testing.assert_close(res_inputs[name], inputs[name], rtol=0, atol=0)

    ref_time = compute_ref_performance(**fresh(inputs))
    res_time = compute_res_performance(**fresh(inputs))
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
