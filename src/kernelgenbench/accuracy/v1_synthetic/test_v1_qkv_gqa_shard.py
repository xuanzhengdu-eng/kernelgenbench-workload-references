"""Accuracy and performance for v1_qkv_gqa_shard (DeepSpeed fused QKV sharding).

Item: row 312 (OP-5c176442ee63).  One logical entry, ``shard_qkv_param``, with
three argument branches: MHA (``n_heads_q == n_heads_kv`` or heads derived from
the fused width), GQA with even KV sharding (``n_kv_heads >= num_shards``) and
GQA with KV replication (``n_kv_heads < num_shards``).  ``compute_ref`` loads
this batch's reference through ``kernelgenbench.baseline``; ``compute_res``
calls the official pinned operator that the verifier bound as the Solution.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from sandbox.utils.accuracy_utils import CustomBenchmarkResult

# (num_shards, shard_rank, head_size, n_heads_q or None, n_heads_kv or None,
#  bias instead of weight, strided weight)
SHAPES = [
    (2, 0, 8, 8, 4, False, False),   # even KV sharding, weight
    (2, 1, 8, 8, 4, True, False),    # even KV sharding, bias
    (4, 3, 8, 8, 2, True, False),    # KV replication, last shard
    (4, 0, 8, 8, 2, False, False),   # KV replication, first shard
    (8, 7, 8, 8, 2, False, False),   # KV replication, degree 4
    (2, 1, 8, 4, 4, False, False),   # MHA
    (2, 0, 8, None, None, False, False),  # MHA, heads derived from the width
    (1, 0, 8, 8, 2, False, False),   # no sharding
    (2, 1, 8, 8, 4, False, True),    # strided weight (transposed layout)
]
COMBOS = [(spec, dtype) for spec in SHAPES
          for dtype in (torch.float32, torch.bfloat16)]


def input_build(config):
    spec, dtype = config
    world, rank, head, nq, nkv, bias, strided = spec
    if nq is None:
        # The fused width must be 3 * heads * head_size for the derived branch.
        nq = 6
        nkv = 6
        rows = 3 * nq * head
    else:
        rows = (nq + 2 * nkv) * head
    generator = torch.Generator(device="cpu").manual_seed(20261115 + rank)
    if bias:
        param = torch.randn((rows,), dtype=dtype, generator=generator)
    elif strided:
        param = torch.randn((32, rows), dtype=dtype, generator=generator).t()
        assert not param.is_contiguous() and param.stride()[0] == 1
    else:
        param = torch.randn((rows, 32), dtype=dtype, generator=generator)
    return {
        "param": param,
        "shard_rank": rank,
        "num_shards": world,
        "head_size": head,
        "n_heads_q": None if spec[3] is None else nq,
        "n_heads_kv": None if spec[3] is None else nkv,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_qkv_gqa_shard(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_qkv_gqa_shard(**inputs)


def compare(ref_out, res_out, dtype):
    assert isinstance(res_out, torch.Tensor), type(res_out).__name__
    assert res_out.shape == ref_out.shape, (res_out.shape, ref_out.shape)
    assert res_out.dtype == ref_out.dtype == dtype
    # Sharding only selects rows, so the values must match exactly.
    torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0)


def _measure(fn, inputs, device, iters=30, warmup=5):
    for _ in range(warmup):
        fn(**inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    if start is not None:
        start.record()
        for _ in range(iters):
            fn(**inputs)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters
    import time
    begin = time.perf_counter()
    for _ in range(iters):
        fn(**inputs)
    return (time.perf_counter() - begin) * 1000 / iters


@label("v1_qkv_gqa_shard")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_qkv_gqa_shard(config):
    spec, dtype = config
    device = kernelgenbench.device
    inputs = {name: value.to(device) if isinstance(value, torch.Tensor) else value
              for name, value in input_build(config).items()}
    ref_inputs = {name: value.clone() if isinstance(value, torch.Tensor) else value
                  for name, value in inputs.items()}
    res_inputs = {name: value.clone() if isinstance(value, torch.Tensor) else value
                  for name, value in inputs.items()}

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    world, rank, head, nq, nkv, bias, _ = spec
    q_heads = 6 if spec[3] is None else nq
    kv_heads = 6 if spec[3] is None else nkv
    if q_heads == kv_heads:
        # MHA shards Q, K and V as three matrices, aligned to the head size.
        per_rank_heads = q_heads // world + (1 if rank < q_heads % world else 0)
        expected_rows = 3 * per_rank_heads * head
    elif kv_heads >= world:
        expected_rows = (q_heads + 2 * kv_heads) * head // world
    else:
        # One replicated KV head plus this shard's Q heads.
        expected_rows = (q_heads // world + 2) * head
    assert res_out.shape[0] == expected_rows, (res_out.shape, expected_rows)
    # The result is a new tensor; the source parameter and its layout survive.
    assert not torch._C._is_alias_of(res_out, res_inputs["param"])
    torch.testing.assert_close(res_inputs["param"], inputs["param"], rtol=0, atol=0)

    ref_time = _measure(compute_ref, ref_inputs, device)
    res_time = _measure(compute_res, res_inputs, device)

    # Illegal argument combinations raise on both sides: KV heads without Q
    # heads, and fewer query heads than shards.
    if spec == SHAPES[0]:
        bad = {"param": inputs["param"], "shard_rank": 0, "num_shards": 2,
               "head_size": 8, "n_heads_q": None, "n_heads_kv": 2}
        for side in (compute_ref, compute_res):
            with pytest.raises(ValueError):
                side(**bad)
        small = {"param": torch.randn(3 * 8), "shard_rank": 1, "num_shards": 2,
                 "head_size": 8, "n_heads_q": 1, "n_heads_kv": 1}
        for side in (compute_ref, compute_res):
            with pytest.raises(ValueError):
                side(**small)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
