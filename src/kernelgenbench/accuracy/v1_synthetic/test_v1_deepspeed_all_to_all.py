"""Accuracy and performance for v1_deepspeed_all_to_all.

Row 548 of the original 723-item table, item OP-c00325593c62-DS.  The Solution
is the real pinned DeepSpeed ``_AllToAll`` from ``deepspeed.moe.sharded_moe``;
the reference states the same equal-split contract.

Contract under test
-------------------
* one argument (no split lists): every rank contributes the same shape;
* the output is always a fresh ``empty_like`` tensor, even for a single rank, so
  it never aliases the input;
* the equal split copies rank-major blocks: position ``i`` of the output is row
  ``i % (rows / world)`` of rank ``i // (rows / world)``;
* the input is only read; the single-rank case still allocates and copies.

COMBOS axes: (world size, dtype, layout), with ``strided`` exercising the
``contiguous()`` copy.  The backward arity (two elements) is covered by the
paired backward item.
"""
import json
import os
import tempfile
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

import kernelgenbench


# (world size, dtype, layout)
COMBOS = [
    (1, torch.float32, "contiguous"),
    (2, torch.float32, "contiguous"),
    (2, torch.int64, "strided"),
]

ROWS = 4
WIDTH = 3


def input_build(config, rank=0):
    """Per-rank input: ``[ROWS, WIDTH]`` with ``rank * 100`` added, CPU."""
    world, dtype, layout = config
    base = torch.arange(ROWS * WIDTH, dtype=torch.int64).reshape(ROWS, WIDTH)
    values = ((base + rank * 100) % 10000).to(dtype)
    if layout == "strided":
        wide = torch.zeros((ROWS, WIDTH * 2), dtype=dtype)
        wide[:, ::2] = values
        values = wide[:, ::2]
        assert not values.is_contiguous()
    return {"input_": values}


def oracle(config, rank):
    world, dtype, _ = config
    per_rank = ROWS // world
    pieces = [input_build(config, source)["input_"].contiguous()[
        rank * per_rank:(rank + 1) * per_rank] for source in range(world)]
    return torch.cat(pieces, dim=0).to(dtype)


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_deepspeed_all_to_all(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_deepspeed_all_to_all(**inputs)


def compare(ref_out, res_out, config, inputs):
    world, dtype, _ = config
    expected = oracle(config, dist.get_rank(dist.group.WORLD))
    for name, out in (("reference", ref_out), ("solution", res_out)):
        assert isinstance(out, torch.Tensor), name
        assert out.shape == expected.shape
        assert out.dtype == dtype
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        assert not torch._C._is_alias_of(out, inputs["input_"]), \
            f"{name} must return a fresh tensor even for one rank"
        assert out.is_contiguous()


def worker(rank, config, rendezvous, outdir):
    torch.set_num_threads(1)
    import datetime
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank,
                            world_size=config[0],
                            timeout=datetime.timedelta(seconds=120))
    try:
        inputs = input_build(config, rank)
        ref_out = compute_ref(**inputs, group=dist.group.WORLD)
        res_out = compute_res(**inputs, group=dist.group.WORLD)
        compare(ref_out, res_out, config, inputs)
        saved = inputs["input_"].clone()

        def timed(fn, repeats=5):
            arguments = input_build(config, rank)
            for _ in range(2):
                fn(**arguments, group=dist.group.WORLD)
            samples = []
            for _ in range(repeats):
                arguments = input_build(config, rank)
                dist.barrier()
                begin = time.perf_counter()
                fn(**arguments, group=dist.group.WORLD)
                dist.barrier()
                samples.append((time.perf_counter() - begin) * 1000)
            samples.sort()
            return samples[len(samples) // 2]

        ref_time = timed(compute_ref)
        res_time = timed(compute_res)
        torch.testing.assert_close(inputs["input_"], saved, rtol=0, atol=0)
        with open(os.path.join(outdir, f"{rank}.json"), "w", encoding="utf8") as handle:
            json.dump([ref_time, res_time], handle)
    finally:
        dist.destroy_process_group()


@label("v1_deepspeed_all_to_all")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_deepspeed_all_to_all(config):
    with tempfile.TemporaryDirectory(prefix="v1-ds-alltoall-") as directory:
        mp.start_processes(worker,
                           args=(config, os.path.join(directory, "rdzv"), directory),
                           nprocs=config[0], join=True, start_method="fork")
        times = [json.load(open(os.path.join(directory, f"{rank}.json"),
                                encoding="utf8")) for rank in range(config[0])]
    ref_time = max(entry[0] for entry in times)
    res_time = max(entry[1] for entry in times)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
