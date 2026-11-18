"""Accuracy and performance for v1_megatron_all_to_all_backward.

Row 547 of the original 723-item table, item OP-6e5857564a8d-MG.  The Solution is
the real pinned Megatron ``_AllToAll.backward``; the reference states the same
reverse-exchange contract.

Contract under test
-------------------
The backward of an all-to-all is the all-to-all with the split lists swapped.
Because ``_AllToAll`` was declared with four inputs
``(group, input, output_split_sizes, input_split_sizes)``, the backward returns a
four-element tuple: ``(None, grad_input, None, None)``.  ``group`` and both split
lists are non-differentiable metadata, so only slot 1 carries a tensor.  With one
rank the forward returned a view of the input, so the gradient passes through and
the tuple is still four elements long.

The test drives the real autograd path (``y = all_to_all(x); y.backward(g)``)
and compares the input gradient with the reference, and additionally calls the
pinned ``backward`` directly with a context carrying the group/split metadata to
assert the tuple arity.

COMBOS axes: (world size, dtype).  The exchange is a permutation of rows, so it
is exact for integer dtypes as well.
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

from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

import kernelgenbench


# (world size, dtype)
COMBOS = [
    (1, torch.float32),
    (1, torch.int64),
    (2, torch.float32),
    (2, torch.int64),
]

ROWS = 4
WIDTH = 3


def input_build(config, rank=0):
    """Per-rank upstream input and incoming gradient, both ``[ROWS, WIDTH]``."""
    world, dtype = config
    base = torch.arange(ROWS * WIDTH, dtype=torch.int64).reshape(ROWS, WIDTH)
    forward_input = ((base + rank * 100) % 10000).to(dtype)
    grad_output = ((base + rank * 7 + 3) % 10000).to(dtype)
    return {"input_": forward_input, "grad_output": grad_output}


def oracle(config, rank):
    """Expected gradient of the forward all-to-all for this rank."""
    world, dtype = config
    if world == 1:
        return input_build(config, rank)["grad_output"]
    per_rank = ROWS // world
    pieces = [input_build(config, source)["grad_output"][
        rank * per_rank:(rank + 1) * per_rank] for source in range(world)]
    return torch.cat(pieces, dim=0).to(dtype)


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_megatron_all_to_all_backward(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_megatron_all_to_all_backward(**inputs)


def compare(ref_out, res_out, config, rank, group):
    world, dtype = config
    assert isinstance(ref_out, tuple) and len(ref_out) == 4, ref_out
    assert isinstance(res_out, tuple) and len(res_out) == 4, res_out
    assert ref_out[0] is None and res_out[0] is None
    assert all(value is None for value in ref_out[2:]), "metadata slots must be None"
    assert all(value is None for value in res_out[2:]), "metadata slots must be None"
    expected = oracle(config, rank)
    for name, grad in (("reference", ref_out[1]), ("solution", res_out[1])):
        assert isinstance(grad, torch.Tensor), name
        assert grad.shape == expected.shape and grad.dtype == dtype
        torch.testing.assert_close(grad, expected, rtol=0, atol=0)
    if world > 1:
        assert not torch._C._is_alias_of(res_out[1], grad_output_of(config, rank))


def grad_output_of(config, rank):
    return input_build(config, rank)["grad_output"]


def worker(rank, config, rendezvous, outdir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank,
                            world_size=config[0],
                            timeout=datetime.timedelta(seconds=120))
    group = dist.group.WORLD
    try:
        inputs = input_build(config, rank)
        tensors = {"group": group, "grad_output": inputs["grad_output"]}
        ref_out = compute_ref(**tensors)
        res_out = compute_res(**tensors)
        compare(ref_out, res_out, config, rank, group)

        # Real autograd path: the gradient of the official forward must equal
        # the reference backward.  Integer dtypes cannot require grad, so that
        # half of the check only runs for floating point configs.
        if not inputs["input_"].is_floating_point():
            with open(os.path.join(outdir, f"{rank}.json"), "w", encoding="utf8") as handle:
                json.dump([0.0, 0.0], handle)
            return
        forward_input = inputs["input_"].clone().requires_grad_(True)
        upstream = kernelgenbench.solution.v1_megatron_all_to_all(
            group, forward_input)
        upstream.backward(inputs["grad_output"])
        torch.testing.assert_close(forward_input.grad, oracle(config, rank),
                                   rtol=0, atol=0)

        def timed(fn, repeats=5):
            for _ in range(2):
                fn(**tensors)
            samples = []
            for _ in range(repeats):
                dist.barrier()
                begin = time.perf_counter()
                fn(**tensors)
                dist.barrier()
                samples.append((time.perf_counter() - begin) * 1000)
            samples.sort()
            return samples[len(samples) // 2]

        ref_time = timed(compute_ref)
        res_time = timed(compute_res)
        with open(os.path.join(outdir, f"{rank}.json"), "w", encoding="utf8") as handle:
            json.dump([ref_time, res_time], handle)
    finally:
        dist.destroy_process_group()


@label("v1_megatron_all_to_all_backward")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_all_to_all_backward(config):
    with tempfile.TemporaryDirectory(prefix="v1-alltoall-backward-") as directory:
        mp.start_processes(worker,
                           args=(config, os.path.join(directory, "rdzv"), directory),
                           nprocs=config[0], join=True, start_method="fork")
        times = [json.load(open(os.path.join(directory, f"{rank}.json"),
                                encoding="utf8")) for rank in range(config[0])]
    ref_time = max(entry[0] for entry in times)
    res_time = max(entry[1] for entry in times)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
