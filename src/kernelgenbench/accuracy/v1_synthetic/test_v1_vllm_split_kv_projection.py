"""Accuracy and performance for v1_vllm_split_kv_projection.

Input: the MLA `kv_b_proj` weight (num_heads*(qk_nope+v), kv_lora_rank), the
requested output dtype and the head geometry. The upstream unquantized branch
casts the weight, transposes it and returns two views:
W_UK [rank, heads, qk_nope] and W_UV [rank, heads, v].

Key assertions: tuple length, exact shapes/dtypes, bitwise values against the
official helper, identical strides, and identical storage aliasing with the
input weight (the cast is free only when out_dtype already matches).
"""
import statistics

import pytest
import torch

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

NAME = "v1_vllm_split_kv_projection"

# (heads, qk_nope width, v width, low-rank width, weight dtype, output dtype)
COMBOS = [
    (1, 4, 3, 8, torch.float32, torch.float32),
    (2, 4, 3, 8, torch.bfloat16, torch.bfloat16),
    (3, 2, 5, 7, torch.bfloat16, torch.float32),
    (4, 8, 8, 4, torch.float16, torch.bfloat16),
]


def input_build(config):
    heads, qk, v, rank, dtype, out_dtype = config
    generator = torch.Generator().manual_seed(612)
    weight = torch.randn((heads * (qk + v), rank), dtype=dtype, generator=generator)
    return dict(weight=weight, out_dtype=out_dtype, kv_lora_rank=rank,
                num_heads=heads, qk_nope_head_dim=qk, v_head_dim=v)


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_vllm_split_kv_projection(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_vllm_split_kv_projection(**inputs)


def compare(ref_out, res_out, ref_weight, res_weight, config):
    heads, qk, v, rank, _, out_dtype = config
    assert isinstance(ref_out, tuple) and isinstance(res_out, tuple)
    assert len(ref_out) == len(res_out) == 2
    for expected, actual in zip(ref_out, res_out):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == expected.dtype == out_dtype
        assert actual.shape == expected.shape
        assert actual.stride() == expected.stride()
        assert actual.storage_offset() == expected.storage_offset()
        assert torch.equal(actual, expected), "projection values differ"
        assert (torch._C._is_alias_of(actual, res_weight)
                == torch._C._is_alias_of(expected, ref_weight)), \
            "storage aliasing with the input weight differs"
        assert torch._C._is_alias_of(actual, res_weight) == (config[4] == out_dtype), \
            "output must alias the weight exactly when no dtype cast happens"
    assert ref_out[0].shape == (rank, heads, qk)
    assert ref_out[1].shape == (rank, heads, v)


def _bench(fn, inputs):
    device = kernelgenbench.device
    prepared = dict(inputs)
    prepared["weight"] = inputs["weight"].to(device)
    for _ in range(5):
        fn(**prepared)
    torch.cuda.synchronize()
    samples = []
    for _ in range(20):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(**prepared)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def compute_ref_performance(inputs):
    return _bench(compute_ref, inputs)


def compute_res_performance(inputs):
    return _bench(compute_res, inputs)


@label("v1_vllm_split_kv_projection")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_split_kv_projection(config):
    inputs = input_build(config)
    ref_weight = inputs["weight"].clone()
    res_weight = inputs["weight"].clone()
    ref_inputs = dict(inputs, weight=ref_weight)
    res_inputs = dict(inputs, weight=res_weight)
    before = inputs["weight"].clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, ref_weight, res_weight, config)
    assert torch.equal(before, ref_weight) and torch.equal(before, res_weight)

    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
