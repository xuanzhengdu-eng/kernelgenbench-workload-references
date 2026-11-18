"""Accuracy and performance for v1_megatron_gather_reduce_scatter_backward.

Megatron first-axis gather backward (pinned 266f1c97).  Each case runs a real
Gloo process group: every rank builds its own gradient, the test checks the
summed-and-scattered segment against an independent all-reduce+slice oracle, the
5-tuple return, the one-rank identity bypass, contiguity/aliasing and that the
incoming gradient was not modified.
"""
from kernelgenbench.accuracy.v1_synthetic.pinned_collective_cuda import encode_candidate, bind_candidate
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


# (world size, non-contiguous gradient)
COMBOS = [(1, False), (2, False), (2, True), (3, False), (3, True)]


def input_build(config, rank, dtype=torch.float32):
    world, noncontiguous = config
    rows, width = 6, 4
    values = (torch.arange(rows * width, dtype=torch.float32, device="cuda").reshape(rows, width) + rank * 10).to(device="cuda", dtype=dtype)
    if noncontiguous:
        storage = torch.empty((rows, width * 2), dtype=dtype, device="cuda")
        storage[:, ::2] = values
        values = storage[:, ::2]
    return {"grad_output": values}


def compute_ref(grad_output, group):
    return kernelgenbench.baseline.v1_megatron_gather_reduce_scatter_backward(grad_output, group)


def compute_res(grad_output, group):
    return kernelgenbench.solution.v1_megatron_gather_reduce_scatter_backward(grad_output, group)


def expected_grad(config, rank, dtype):
    world = config[0]
    total = sum(input_build(config, other, dtype)["grad_output"] for other in range(world))
    size = total.shape[0] // world
    return total.narrow(0, rank * size, size).clone(memory_format=torch.contiguous_format)


def compare(expected, outputs, config, rank, source):
    world = config[0]
    assert isinstance(outputs, tuple) and len(outputs) == 5, (
        f"rank {rank}: the official backward returns a 5-tuple, got {type(outputs)}"
    )
    grad = outputs[0]
    assert all(value is None for value in outputs[1:]), "non-gradient returns must be None"
    assert grad.shape == expected.shape and grad.dtype == expected.dtype
    torch.testing.assert_close(grad, expected, rtol=0, atol=0)
    if world == 1:
        # Official one-rank bypass returns the input object unchanged.
        assert grad is source, f"rank {rank}: one-rank bypass must return the input"
    else:
        assert grad.is_contiguous(), f"rank {rank}: the shard must be contiguous"
        assert not torch._C._is_alias_of(grad, source), f"rank {rank}: the shard must be fresh"


def _timed_ms(fn, config, dtype, rank, repeats=3):
    dist.barrier()
    start = time.perf_counter()
    for _ in range(repeats):
        fn(**input_build(config, rank, dtype), group=dist.group.WORLD)
    dist.barrier()
    return (time.perf_counter() - start) * 1000 / repeats


def compute_ref_performance(config, dtype, rank):
    return _timed_ms(compute_ref, config, dtype, rank)


def compute_res_performance(config, dtype, rank):
    return _timed_ms(compute_res, config, dtype, rank)


def _bind_worker_solution():
    """Bind the Solution namespace in a fresh worker (spawn re-imports the package)."""
    target = os.environ.get("KGB_WORKER_SOLUTION")
    if not target:
        return
    import importlib
    import sys
    adapter_dir = os.environ.get("KGB_ADAPTER_DIR")
    if adapter_dir and adapter_dir not in sys.path:
        sys.path.insert(0, adapter_dir)
    module_name, _, attribute = target.partition(":")
    kernelgenbench.triton = getattr(importlib.import_module(module_name), attribute)


def _worker(rank, config, dtype, path, outdir, encoded):
    torch.cuda.set_device(rank)
    bind_candidate(encoded, 'v1_megatron_gather_reduce_scatter_backward')
    torch.set_num_threads(1)
    dist.init_process_group("nccl", init_method="file://" + path, rank=rank,
                            world_size=config[0], timeout=datetime.timedelta(seconds=60))
    try:
        arguments = input_build(config, rank, dtype)
        ref_side = {"grad_output": arguments["grad_output"].clone()}
        res_side = {"grad_output": arguments["grad_output"].clone()}
        before = arguments["grad_output"].clone()
        ref_out = compute_ref(**ref_side, group=dist.group.WORLD)
        res_out = compute_res(**res_side, group=dist.group.WORLD)
        expected = expected_grad(config, rank, dtype)
        compare(expected, ref_out, config, rank, ref_side["grad_output"])
        compare(expected, res_out, config, rank, res_side["grad_output"])
        assert torch.equal(ref_out[0], res_out[0]), f"rank {rank}: reference and solution disagree"
        assert torch.equal(ref_side["grad_output"], before), "reference modified the gradient"
        assert torch.equal(res_side["grad_output"], before), "solution modified the gradient"
        timings = [compute_ref_performance(config, dtype, rank),
                   compute_res_performance(config, dtype, rank)]
        with open(os.path.join(outdir, f"{rank}.json"), "w", encoding="utf8") as handle:
            json.dump(timings, handle)
    finally:
        dist.destroy_process_group()


@label("v1_megatron_gather_reduce_scatter_backward")
@parametrize("config", COMBOS)
@parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("config", COMBOS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_accuracy_v1_megatron_gather_reduce_scatter_backward(config, dtype):
    with tempfile.TemporaryDirectory(prefix="v1-megatron-gather-rs-") as directory:
        mp.start_processes(
            _worker,
            args=(config, dtype, os.path.join(directory, "rdzv"), directory, encode_candidate('v1_megatron_gather_reduce_scatter_backward')),
            nprocs=config[0],
            join=True,
            start_method="spawn",
        )
        timings = [json.load(open(os.path.join(directory, f"{rank}.json"), encoding="utf8"))
                   for rank in range(config[0])]
    ref_time = max(item[0] for item in timings)
    res_time = max(item[1] for item in timings)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
