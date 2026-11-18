"""Accuracy and performance for DeepSpeed vocabulary-parallel cross entropy.

Every rank builds one local vocabulary shard, so concatenating the rank shards
reconstructs the full logits and gives an independent ``F.cross_entropy``
oracle.  The candidate is bound to the official DeepSpeed
``vocab_parallel_cross_entropy`` with the real Gloo process group.  The test
covers one-rank and two-rank shards, a target on another rank, an
``ignore_index`` row and a BF16 input.

The per-rank worker only orchestrates: the input construction, both sides of
the comparison and both timings are the module-level functions below, so the
file reads like every other test in this batch.
"""
import datetime
import json
import os
import tempfile
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (world size, rows, local vocab, input dtype, label smoothing)
COMBOS = [
    (1, 3, 8, torch.float32, 0.0),
    (2, 3, 8, torch.float32, 0.0),
    (2, 4, 8, torch.bfloat16, 0.0),
]


def input_build(config, rank):
    world, rows, n, dtype, smoothing = config
    generator = torch.Generator().manual_seed(20261110 + rank)
    logits = torch.randn((rows, n), generator=generator).to(dtype)
    target = torch.tensor([i * (world * n - 1) // max(rows - 1, 1) for i in range(rows)],
                          dtype=torch.int64)
    target[1] = -100
    return dict(logits=logits, target=target, label_smoothing=smoothing,
                rank=rank, world_size=world, backend="deepspeed", ignore_index=-100)


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_megatron_vocab_parallel_ce_deepspeed(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_megatron_vocab_parallel_ce_deepspeed(**inputs)


def _oracle(config):
    world, rows, n, dtype, _smoothing = config
    joined = torch.cat([input_build(config, rank)["logits"].float()
                        for rank in range(world)], dim=-1)
    target = input_build(config, 0)["target"].clone()
    return joined, torch.nn.functional.cross_entropy(
        joined, target, reduction="none", ignore_index=-100)


def compare(ref_out, res_out, config):
    _joined, oracle = _oracle(config)
    assert ref_out.shape == res_out.shape == (config[1],)
    assert ref_out.dtype == res_out.dtype == torch.float32
    torch.testing.assert_close(res_out, ref_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(ref_out, oracle, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(res_out, oracle, rtol=1e-5, atol=1e-5)


def _sync(group):
    # CPU all-reduce as the barrier: dist.barrier() would pick a CUDA device in
    # this fork-based CPU harness, which cannot initialize in that child.
    dist.all_reduce(torch.zeros(1), group=group)


def compute_ref_performance(config, group):
    base = input_build(config, dist.get_rank(group))
    _sync(group)
    start = time.perf_counter()
    for _ in range(2):
        arguments = dict(base)
        arguments["logits"] = base["logits"].clone()
        compute_ref(**arguments, group=group)
    _sync(group)
    return (time.perf_counter() - start) * 500.0


def compute_res_performance(config, group):
    base = input_build(config, dist.get_rank(group))
    _sync(group)
    start = time.perf_counter()
    for _ in range(2):
        arguments = dict(base)
        arguments["logits"] = base["logits"].clone()
        compute_res(**arguments, group=group)
    _sync(group)
    return (time.perf_counter() - start) * 500.0


def _worker(rank, config, path, outdir):
    torch.set_num_threads(1)
    # DeepSpeed's pg_utils reads its own comm backend, so the group must be
    # created through deepspeed.init_distributed.  Imported lazily so other
    # framework containers can import this module without DeepSpeed.
    try:
        import deepspeed
    except ImportError:
        dist.init_process_group("gloo", init_method="file://" + path, rank=rank,
                                world_size=config[0],
                                timeout=datetime.timedelta(seconds=60))
        destroy = dist.destroy_process_group
    else:
        deepspeed.init_distributed(dist_backend="gloo", init_method="file://" + path,
                                   rank=rank, world_size=config[0],
                                   auto_mpi_discovery=False, verbose=False,
                                   timeout=datetime.timedelta(seconds=60))
        destroy = deepspeed.comm.destroy_process_group
    try:
        ref_inputs = input_build(config, rank)
        res_inputs = input_build(config, rank)
        before = ref_inputs["logits"].clone()
        ref_out = compute_ref(**ref_inputs, group=dist.group.WORLD)
        res_out = compute_res(**res_inputs, group=dist.group.WORLD)
        compare(ref_out, res_out, config)

        torch.testing.assert_close(ref_inputs["logits"], before, rtol=0, atol=0)
        torch.testing.assert_close(res_inputs["logits"], before, rtol=0, atol=0)
        assert torch.equal(ref_inputs["target"], res_inputs["target"])

        ref_time = compute_ref_performance(config, dist.group.WORLD)
        res_time = compute_res_performance(config, dist.group.WORLD)
        with open(os.path.join(outdir, str(rank) + ".json"), "w") as handle:
            json.dump([ref_time, res_time], handle)
    finally:
        destroy()


@label("v1_megatron_vocab_parallel_ce_deepspeed")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_vocab_parallel_ce_deepspeed(config):
    with tempfile.TemporaryDirectory(prefix="v1-vocab-ce-ds-") as directory:
        mp.start_processes(_worker,
                           args=(config, os.path.join(directory, "rdzv"), directory),
                           nprocs=config[0], join=True, start_method="fork")
        times = [json.load(open(os.path.join(directory, str(rank) + ".json")))
                 for rank in range(config[0])]
    ref_time = max(value[0] for value in times)
    res_time = max(value[1] for value in times)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
