"""Accuracy and performance for v1_attn_out_gqa_shard (DeepSpeed, pinned 8af59b2e).

This is the GQA branch of the official ``shard_attn_out_param``, directly
callable with explicit shard configuration (no process group).  Inputs are built
on CPU and copied to the device named in each combination; the official
function is device-agnostic, so both ``cpu`` and ``cuda`` are covered.

Contract asserted per case:
  * 1-D bias -> the parameter object itself on rank 0, ``None`` on other ranks;
  * 2-D weight -> ``param.shape[0] == head_size * n_heads_q`` with
    ``n_heads_q != n_heads_kv``, then a head-aligned inner-dimension shard;
  * when ``n_heads_kv % num_shards == 0`` the whole inner dimension is sharded;
  * otherwise ``num_shards % n_heads_kv == 0``, this rank picks its KV head
    group first and is sharded inside that group;
  * an unsharded parameter (``num_shards == 1``) is returned as the same object;
  * the input parameter is never modified.
"""
import statistics
import time

import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.bias_activation_test_utils import fresh
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (n_heads_q, n_heads_kv, num_shards, shard_rank, head_size, is_bias)
# The weight shape is (n_heads_q * head_size, n_heads_q * head_size) and the
# bias shape is (n_heads_q * head_size,).
CASES = [
    (8, 4, 2, 0, 8, False),  # even KV sharding, rank 0
    (8, 4, 2, 1, 8, False),  # even KV sharding, rank 1
    (8, 2, 4, 0, 8, False),  # 4 shards over 2 KV heads: KV group 0, local 0
    (8, 2, 4, 3, 8, False),  # 4 shards over 2 KV heads: KV group 1, local 1
    (8, 2, 4, 0, 8, True),   # bias on rank 0 -> the same tensor object
    (8, 2, 4, 1, 8, True),   # bias on rank 1 -> None
    (8, 2, 1, 0, 8, False),  # unsharded -> the same tensor object
]

COMBOS = [
    (case, dtype, device)
    for case in CASES
    for dtype in (torch.float32, torch.bfloat16)
    for device in ("cpu", "cuda")
    if device == "cpu" or torch.cuda.is_available()
]


def input_build(config):
    case, dtype, device = config
    n_heads_q, n_heads_kv, num_shards, shard_rank, head_size, is_bias = case
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    width = n_heads_q * head_size
    shape = (width,) if is_bias else (width, width)
    return {
        "param": torch.randn(shape, dtype=dtype, generator=generator, device="cpu"),
        "shard_rank": shard_rank,
        "num_shards": num_shards,
        "head_size": head_size,
        "n_heads_q": n_heads_q,
        "n_heads_kv": n_heads_kv,
    }


def compute_ref(param, shard_rank, num_shards, head_size, n_heads_q, n_heads_kv):
    return kernelgenbench.baseline.v1_attn_out_gqa_shard(
        param, shard_rank, num_shards, head_size, n_heads_q, n_heads_kv
    )


def compute_res(param, shard_rank, num_shards, head_size, n_heads_q, n_heads_kv):
    return kernelgenbench.solution.v1_attn_out_gqa_shard(
        param, shard_rank, num_shards, head_size, n_heads_q, n_heads_kv
    )


def expected_output(source, case):
    """Independent construction of the GQA shard (mirrors the official branch)."""
    n_heads_q, n_heads_kv, num_shards, shard_rank, head_size, is_bias = case
    if is_bias:
        # The bias is added on rank 0 only, so it is not sharded.
        return source if shard_rank == 0 else None
    assert n_heads_q != n_heads_kv, "this file covers only the GQA branch"
    assert source.shape[0] == head_size * n_heads_q
    if num_shards == 1:
        return source
    if n_heads_kv % num_shards == 0:
        group, local_rank, local_shards = source, shard_rank, num_shards
    else:
        assert num_shards % n_heads_kv == 0
        degree = num_shards // n_heads_kv
        kv_head = shard_rank // degree
        local_rank = shard_rank % degree
        factor = n_heads_q // n_heads_kv
        group = source[..., factor * kv_head * head_size:
                              factor * (kv_head + 1) * head_size]
        local_shards = degree
    width = group.shape[-1]
    chunk = width // local_shards
    start = local_rank * chunk
    return group[..., start:start + chunk]


def compare(ref_out, res_out, ref_source, res_source, config, dtype):
    case, device = config[0], config[2]
    is_bias = case[5]
    num_shards = case[2]
    assert (ref_out is None) == (res_out is None), "reference and solution disagree on None"
    if ref_out is not None:
        assert torch.equal(ref_out, res_out), "reference and solution values differ"

    for name, output, source in (
        ("reference", ref_out, ref_source),
        ("solution", res_out, res_source),
    ):
        expected = expected_output(source.detach(), case)
        if expected is None:
            assert output is None, f"{name} must return None on a non-zero shard rank"
            continue
        assert isinstance(output, torch.Tensor), f"{name} must return a tensor"
        assert output.shape == expected.shape, \
            f"{name} shape {tuple(output.shape)} != {tuple(expected.shape)}"
        assert output.dtype == dtype, f"{name} dtype {output.dtype} != {dtype}"
        assert output.device.type == device, f"{name} device {output.device} != {device}"
        # Pure slicing: bit-exact, no tolerance needed.
        assert torch.equal(output, expected.to(device)), \
            f"{name} differs from the independent GQA shard"
        if is_bias or num_shards == 1:
            # Official returns the parameter object itself for unsharded cases.
            assert output is source, f"{name} must return the input object unsharded"
        else:
            assert not torch._C._is_alias_of(output, source), \
                f"{name} aliases the parameter; the shard must be a fresh tensor"


def _measure(fn, inputs):
    """Median single-call milliseconds on the input's own device."""
    device = inputs["param"].device
    samples = []
    for _ in range(5):
        arguments = fresh(inputs, device)
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn(**arguments)
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end)
        else:
            start = time.perf_counter()
            fn(**arguments)
            elapsed = (time.perf_counter() - start) * 1000
        samples.append(elapsed)
    return statistics.median(samples[1:])


def compute_ref_performance(param, shard_rank, num_shards, head_size, n_heads_q, n_heads_kv):
    return _measure(compute_ref, {
        "param": param, "shard_rank": shard_rank, "num_shards": num_shards,
        "head_size": head_size, "n_heads_q": n_heads_q, "n_heads_kv": n_heads_kv,
    })


def compute_res_performance(param, shard_rank, num_shards, head_size, n_heads_q, n_heads_kv):
    return _measure(compute_res, {
        "param": param, "shard_rank": shard_rank, "num_shards": num_shards,
        "head_size": head_size, "n_heads_q": n_heads_q, "n_heads_kv": n_heads_kv,
    })


@label("v1_attn_out_gqa_shard")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_attn_out_gqa_shard(config):
    case, dtype, device = config
    target = torch.device(device)
    inputs = input_build(config)
    ref_inputs = fresh(inputs, target)
    res_inputs = fresh(inputs, target)
    for side in (ref_inputs, res_inputs):
        assert side["param"].dtype == dtype
        assert side["param"].device.type == device
        assert side["param"].shape == inputs["param"].shape

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, ref_inputs["param"], res_inputs["param"], config, dtype)

    # The parameter is read-only for sharding.
    for side in (ref_inputs, res_inputs):
        torch.testing.assert_close(side["param"].cpu(), inputs["param"], rtol=0, atol=0)

    perf_inputs = fresh(inputs, target)
    ref_time = compute_ref_performance(**perf_inputs)
    res_time = compute_res_performance(**perf_inputs)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time,
        speedup=ref_time / res_time if res_time else float("inf"),
    )
