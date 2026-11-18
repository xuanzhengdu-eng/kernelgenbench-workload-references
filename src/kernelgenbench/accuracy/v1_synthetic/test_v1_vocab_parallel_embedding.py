"""Accuracy and performance for v1_vocab_parallel_embedding (723-row 470).

Operator: Megatron-LM ``VocabParallelEmbedding.forward`` (plain TP all-reduce).
Official source (pinned commit 266f1c97dca477ca8b92c16087506da2000b0b84):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/layers.py#L276-L303

(Same sub-contract in the pinned vLLM and SGLang modules; those wrappers need a
full distributed config, so the Megatron class is what this test drives.)

Contract: every rank holds the contiguous shard
``[rank*local_rows, (rank+1)*local_rows)`` of the vocabulary.  Ids outside the
local range are masked to index 0, their contribution is zeroed, then the full
embedding is all-reduced, so all peers see the same [.., D] tensor.  With
tp_size == 1 upstream skips masking entirely and an out-of-range id raises
``IndexError`` from the gather instead of producing zeros.
"""
import datetime
import json
import os
import tempfile
import time

import pytest
import torch

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (world, batch, sequence, local vocab rows, width, dtype)
CASES = [
    (1, 2, 5, 8, 7, torch.float32),
    (1, 1, 4, 6, 3, torch.bfloat16),
    (2, 2, 4, 8, 6, torch.float32),
    (2, 1, 5, 6, 8, torch.float32),
    (2, 3, 4, 4, 3, torch.bfloat16),
]
COMBOS = list(CASES)


def _table(world, local, dim, dtype):
    """Full vocabulary table, deterministic and identical on every rank."""
    vocab = local * world
    return ((((torch.arange(vocab * dim, dtype=torch.float32) * 13) % 61) - 30) / 16
            ).reshape(vocab, dim).to(dtype)


def get_input(config, rank=0, group=None):
    """Rank-local weight shard plus shared ids; ids cover in-shard and foreign rows."""
    world, batch, seq, local, dim, dtype = config
    vocab = local * world
    table = _table(world, local, dim, dtype)
    ids = ((torch.arange(batch * seq, dtype=torch.int64) * 5 + local - 1) % vocab
           ).reshape(batch, seq)
    return {"ids": ids, "weight": table[rank * local:(rank + 1) * local].clone(),
            "group": group}


def input_build(config):
    return get_input(config, rank=0, group=None)


def oracle(config):
    """Independent truth: full-table lookup of the same ids."""
    world, batch, seq, local, dim, dtype = config
    vocab = local * world
    table = _table(world, local, dim, dtype)
    ids = ((torch.arange(batch * seq, dtype=torch.int64) * 5 + local - 1) % vocab
           ).reshape(batch, seq)
    return table[ids]


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_vocab_parallel_embedding(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_vocab_parallel_embedding(**inputs)


def compare(ref_out, res_out, config, expected):
    world, batch, seq, local, dim, dtype = config
    for name, value in (("reference", ref_out), ("solution", res_out)):
        assert isinstance(value, torch.Tensor)
        assert value.shape == (batch, seq, dim), name
        assert value.dtype == dtype, name
        assert torch.equal(value, expected), f"{name} differs from the full-table lookup"


def _bench(fn, inputs):
    if inputs["group"] is not None:
        import torch.distributed as dist
        dist.barrier(group=inputs["group"])
    start = time.perf_counter()
    for _ in range(5):
        fn(**inputs)
    if inputs["group"] is not None:
        import torch.distributed as dist
        dist.barrier(group=inputs["group"])
    return (time.perf_counter() - start) * 1000.0 / 5


def worker(rank, config, store, outdir):
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + store, rank=rank,
                            world_size=config[0],
                            timeout=datetime.timedelta(seconds=120))
    try:
        ref_inputs = get_input(config, rank, dist.group.WORLD)
        res_inputs = get_input(config, rank, dist.group.WORLD)
        saved = {name: value.clone() for name, value in res_inputs.items()
                 if isinstance(value, torch.Tensor)}
        expected = oracle(config)
        ref_out = compute_ref(**ref_inputs)
        res_out = compute_res(**res_inputs)
        compare(ref_out, res_out, config, expected)
        for name, initial in saved.items():
            assert torch.equal(res_inputs[name], initial)
            assert not torch._C._is_alias_of(res_out, res_inputs[name])
        # Every rank must observe the identical all-reduced result.
        torch.testing.assert_close(res_out, expected, rtol=0, atol=0)
        timings = [_bench(compute_ref, ref_inputs), _bench(compute_res, res_inputs)]
        with open(os.path.join(outdir, f"{rank}.json"), "w") as handle:
            json.dump(timings, handle)
    finally:
        dist.destroy_process_group()


def _tp_worker_entry(rank, config, store, outdir):
    worker(rank, config, store, outdir)


@label("v1_vocab_parallel_embedding")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vocab_parallel_embedding(config):
    world, batch, seq, local, dim, dtype = config
    if world > 1:
        with tempfile.TemporaryDirectory(prefix="vocab-tp-") as tmp:
            import torch.multiprocessing as mp
            mp.start_processes(
                _tp_worker_entry,
                args=(config, os.path.join(tmp, "store"), tmp),
                nprocs=world, join=True, start_method="fork")
            timings = [json.load(open(os.path.join(tmp, f"{rank}.json")))
                       for rank in range(world)]
        ref_time = max(entry[0] for entry in timings)
        res_time = max(entry[1] for entry in timings)
        return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                     speedup=ref_time / res_time)

    inputs = get_input(config, 0, None)
    res_inputs = get_input(config, 0, None)
    saved = {name: value.clone() for name, value in res_inputs.items()
             if isinstance(value, torch.Tensor)}
    expected = oracle(config)
    ref_out = compute_ref(**inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config, expected)
    for name, initial in saved.items():
        assert torch.equal(res_inputs[name], initial)
        assert not torch._C._is_alias_of(res_out, res_inputs[name])

    ref_time = _bench(compute_ref, inputs)
    res_time = _bench(compute_res, res_inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
