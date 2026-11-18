"""Accuracy and CPU performance for v1_causal_conv1d_varlen.

Every case builds packed sequences, an independent cache, optional per-sequence
cache indices and optional initial-state flags.  The key assertions are output
values, the mutated cache slots and untouched metadata/input tensors.
"""
import time

import pytest
import torch
import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

CASES = [(8, 3, 5, 2), (16, 5, 7, 4)]
COMBOS = [
    (case, dtype, indexed, initial_mode, bias_present, activation)
    for case in CASES
    for dtype in (torch.float32, torch.bfloat16)
    for indexed in (False, True)
    for initial_mode in ("none", "mixed", "all")
    for bias_present in (False, True)
    for activation in (None, "silu")
]


def input_build(config):
    (channels, s1, s2, width), dtype, indexed, initial_mode, bias_present, activation = config
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    return {
        "x": torch.randn((channels, s1 + s2), dtype=dtype, generator=generator),
        "weight": torch.randn((channels, width), dtype=dtype, generator=generator),
        "bias": (torch.randn((channels,), dtype=dtype, generator=generator)
                 if bias_present else None),
        "conv_states": torch.randn((3, channels, width - 1), dtype=dtype, generator=generator),
        "query_start_loc": torch.tensor([0, s1, s1 + s2], dtype=torch.int64),
        "cache_indices": (torch.tensor([2, 0], dtype=torch.int64) if indexed else None),
        "has_initial_state": (torch.tensor([True, initial_mode == "all"], dtype=torch.bool)
                              if initial_mode != "none" else None),
        "activation": activation,
    }


def fresh(inputs):
    return {name: value.clone() if isinstance(value, torch.Tensor) else value
            for name, value in inputs.items()}


def compute_ref(x, weight, bias, conv_states, query_start_loc, cache_indices,
                has_initial_state, activation):
    return kernelgenbench.baseline.v1_causal_conv1d_varlen(
        x, weight, bias, conv_states, query_start_loc, cache_indices,
        has_initial_state, activation)


def compute_res(x, weight, bias, conv_states, query_start_loc, cache_indices,
                has_initial_state, activation):
    return kernelgenbench.solution.v1_causal_conv1d_varlen(
        x, weight, bias, conv_states, query_start_loc, cache_indices,
        has_initial_state, activation)


def compare(ref_out, res_out, ref_state, res_state, dtype):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape and res_out.dtype == ref_out.dtype == dtype
    torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0)
    # Selected slots are the stateful contract; untouched slots must also agree.
    assert torch.equal(res_state, ref_state)


def time_cpu(fn, inputs, iters=120):
    state = inputs["conv_states"].clone()
    for _ in range(10):
        inputs["conv_states"].copy_(state)
        fn(**inputs)
    elapsed = 0.0
    for _ in range(iters):
        inputs["conv_states"].copy_(state)
        start = time.perf_counter()
        fn(**inputs)
        elapsed += time.perf_counter() - start
    return elapsed / iters


def compute_ref_performance(x, weight, bias, conv_states, query_start_loc,
                            cache_indices, has_initial_state, activation):
    return time_cpu(compute_ref, {
        "x": x, "weight": weight, "bias": bias, "conv_states": conv_states,
        "query_start_loc": query_start_loc, "cache_indices": cache_indices,
        "has_initial_state": has_initial_state, "activation": activation,
    })


def compute_res_performance(x, weight, bias, conv_states, query_start_loc,
                            cache_indices, has_initial_state, activation):
    return time_cpu(compute_res, {
        "x": x, "weight": weight, "bias": bias, "conv_states": conv_states,
        "query_start_loc": query_start_loc, "cache_indices": cache_indices,
        "has_initial_state": has_initial_state, "activation": activation,
    })


@label("v1_causal_conv1d_varlen")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_causal_conv1d_varlen(config):
    case, dtype, indexed, initial_mode, _, _ = config
    channels, s1, s2, width = case
    inputs = input_build(config)
    ref_inputs = fresh(inputs)
    res_inputs = fresh(inputs)

    assert ref_inputs["x"].shape == (channels, s1 + s2)
    assert ref_inputs["weight"].shape == (channels, width)
    assert ref_inputs["conv_states"].shape == (3, channels, width - 1)
    assert ref_inputs["query_start_loc"].tolist() == [0, s1, s1 + s2]
    assert (ref_inputs["cache_indices"] is not None) == indexed
    assert (ref_inputs["has_initial_state"] is not None) == (initial_mode != "none")

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, ref_inputs["conv_states"], res_inputs["conv_states"], dtype)

    assert ref_out.shape == (channels, s1 + s2)
    for name in ("x", "weight", "bias", "query_start_loc", "cache_indices", "has_initial_state"):
        if inputs[name] is not None:
            torch.testing.assert_close(res_inputs[name], inputs[name], rtol=0, atol=0)

    ref_time = compute_ref_performance(**fresh(inputs))
    res_time = compute_res_performance(**fresh(inputs))
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
