"""Accuracy and performance for v1_megatron_all_to_all.

Row 548 of the original 723-item table, item OP-c00325593c62-MG.  The Solution
is the real pinned Megatron ``_AllToAll`` (through the public ``all_to_all``
wrapper); the reference states the same contract with plain tensor ops.

Contract under test
-------------------
* ``world_size == 1`` returns the input object unchanged (autograd view, no
  collective);
* otherwise the equal split copies rank-major blocks: position ``i`` of the
  output is row ``i % (rows / world)`` of rank ``i // (rows / world)``;
* the result is a fresh contiguous tensor and the input is not modified;
* ``backward`` performs the reverse exchange (checked by the paired backward
  item, not here).

COMBOS axes: (world size, dtype, layout).  ``strided`` builds the input as a
column view of a wider buffer, which exercises the ``contiguous()`` copy the
official function performs.  Unequal transfers are checked on separate GPUs using NCCL.
"""
import json
import os
import tempfile

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
    (2, torch.int64, "contiguous"),
    (2, torch.float32, "strided"),
    (2, torch.int64, "strided"),
    (2, torch.float32, "unequal"),
]

ROWS = 4
WIDTH = 3


def input_build(config, rank=0):
    """Per-rank input: ``[ROWS, WIDTH]`` with ``rank * 100`` added, CPU."""
    world, dtype, layout = config
    assert ROWS % world == 0
    base = torch.arange(ROWS * WIDTH, dtype=torch.int64).reshape(ROWS, WIDTH)
    values = ((base + rank * 100) % 10000).to(dtype)
    if layout == "strided":
        wide = torch.zeros((ROWS, WIDTH * 2), dtype=dtype)
        wide[:, ::2] = values
        values = wide[:, ::2]
        assert not values.is_contiguous()
    if layout == "unequal":
        send_splits = ((1, 3), (2, 2))
        return {"input_": values.cuda(), "input_split_sizes": list(send_splits[rank]),
                "output_split_sizes": [split[rank] for split in send_splits]}
    return {"input_": values}


def oracle(config, rank):
    """Expected equal-split result for this rank."""
    world, dtype, _ = config
    if world == 1:
        return input_build(config, rank)["input_"]
    blocks = [input_build(config, source)["input_"].contiguous() for source in range(world)]
    if config[2] == "unequal":
        splits = ((1, 3), (2, 2))
        return torch.cat([block.narrow(0, sum(split[:rank]), split[rank])
                          for block, split in zip(blocks, splits)])
    per_rank = ROWS // world
    pieces = [block[rank * per_rank:(rank + 1) * per_rank] for block in blocks]
    return torch.cat(pieces, dim=0).to(dtype)


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_megatron_all_to_all(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_megatron_all_to_all(**inputs)


def compare(ref_out, res_out, config, inputs):
    world, dtype, _ = config
    expected = oracle(config, dist.get_rank(dist.group.WORLD))
    for name, out in (("reference", ref_out), ("solution", res_out)):
        assert isinstance(out, torch.Tensor), name
        assert out.shape == expected.shape, (name, out.shape, expected.shape)
        assert out.dtype == dtype, (name, out.dtype, dtype)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        if world == 1:
            assert out.untyped_storage().data_ptr() == \
                inputs["input_"].untyped_storage().data_ptr(), \
                f"{name} must share the input storage on one rank"
        else:
            assert not torch._C._is_alias_of(out, inputs["input_"]), \
                f"{name} must return a fresh tensor"
            assert out.is_contiguous()


def worker(rank, config, rendezvous, outdir, candidate_bytes=None):
    gpu = config[2] == "unequal"
    if gpu:
        import cloudpickle
        from types import SimpleNamespace
        torch.cuda.set_device(rank)
        kernelgenbench.triton = SimpleNamespace(v1_megatron_all_to_all=cloudpickle.loads(candidate_bytes))
    torch.set_num_threads(1)
    dist.init_process_group("nccl" if gpu else "gloo", init_method="file://" + rendezvous, rank=rank,
                            world_size=config[0],
                            timeout=__import__("datetime").timedelta(seconds=120))
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
                import time
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


@label("v1_megatron_all_to_all")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_all_to_all(config):
    with tempfile.TemporaryDirectory(prefix="v1-alltoall-") as directory:
        gpu = config[2] == "unequal"
        candidate_bytes = None
        if gpu:
            import cloudpickle
            if torch.cuda.device_count() < config[0]:
                raise RuntimeError("unequal all-to-all needs a separate visible GPU per rank")
            candidate_bytes = cloudpickle.dumps(kernelgenbench.solution.v1_megatron_all_to_all)
        mp.start_processes(worker,
                           args=(config, os.path.join(directory, "rdzv"), directory, candidate_bytes),
                           nprocs=config[0], join=True, start_method="spawn" if gpu else "fork")
        times = [json.load(open(os.path.join(directory, f"{rank}.json"),
                                encoding="utf8")) for rank in range(config[0])]
    ref_time = max(entry[0] for entry in times)
    res_time = max(entry[1] for entry in times)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
