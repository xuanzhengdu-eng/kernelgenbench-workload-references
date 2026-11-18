"""Accuracy and performance for v1_deepspeed_all_to_all_backward.

Row 547 of the original 723-item table, item OP-6e5857564a8d-DS.  The Solution is
the real pinned DeepSpeed ``_AllToAll.backward``; the reference states the same
contract.

Contract under test
-------------------
DeepSpeed's autograd function has two inputs ``(group, input)``, so its backward
returns exactly ``(None, grad)``: one ``None`` for the group and the gradient of
the input.  The gradient is the same equal-split all-to-all applied to the
incoming gradient, and it is a fresh tensor even for a single rank (the forward
allocated one too, so nothing aliases).

The test drives the real autograd path and additionally calls the pinned
``backward`` with a context to assert the two-element arity.

COMBOS axes: (world size, dtype).
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
    (2, torch.float32),
    (2, torch.int64),
]

ROWS = 4
WIDTH = 3


def input_build(config, rank=0):
    world, dtype = config
    base = torch.arange(ROWS * WIDTH, dtype=torch.int64).reshape(ROWS, WIDTH)
    forward_input = ((base + rank * 100) % 10000).to(dtype)
    grad_output = ((base + rank * 7 + 3) % 10000).to(dtype)
    return {"input_": forward_input, "grad_output": grad_output}


def oracle(config, rank):
    world, dtype = config
    per_rank = ROWS // world
    pieces = [input_build(config, source)["grad_output"][
        rank * per_rank:(rank + 1) * per_rank] for source in range(world)]
    return torch.cat(pieces, dim=0).to(dtype)


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_deepspeed_all_to_all_backward(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_deepspeed_all_to_all_backward(**inputs)


def compare(ref_out, res_out, config, rank):
    world, dtype = config
    for name, out in (("reference", ref_out), ("solution", res_out)):
        assert isinstance(out, tuple) and len(out) == 2, (name, out)
        assert out[0] is None, f"{name} must return None for the group slot"
        assert isinstance(out[1], torch.Tensor)
        assert out[1].shape == (ROWS, WIDTH) and out[1].dtype == dtype
        torch.testing.assert_close(out[1], oracle(config, rank), rtol=0, atol=0)
        assert not torch._C._is_alias_of(out[1], input_build(config, rank)["grad_output"])


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
        compare(ref_out, res_out, config, rank)

        # Integer dtypes cannot require grad; the direct backward call above
        # already covered the value and arity for them.
        if not inputs["input_"].is_floating_point():
            with open(os.path.join(outdir, f"{rank}.json"), "w", encoding="utf8") as handle:
                json.dump([0.0, 0.0], handle)
            return
        forward_input = inputs["input_"].clone().requires_grad_(True)
        upstream = kernelgenbench.solution.v1_deepspeed_all_to_all(group, forward_input)
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


@label("v1_deepspeed_all_to_all_backward")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_deepspeed_all_to_all_backward(config):
    with tempfile.TemporaryDirectory(prefix="v1-ds-alltoall-backward-") as directory:
        mp.start_processes(worker,
                           args=(config, os.path.join(directory, "rdzv"), directory),
                           nprocs=config[0], join=True, start_method="fork")
        times = [json.load(open(os.path.join(directory, f"{rank}.json"),
                                encoding="utf8")) for rank in range(config[0])]
    ref_time = max(entry[0] for entry in times)
    res_time = max(entry[1] for entry in times)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
