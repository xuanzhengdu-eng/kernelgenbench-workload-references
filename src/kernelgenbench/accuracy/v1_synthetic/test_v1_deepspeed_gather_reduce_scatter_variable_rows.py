"""Accuracy and performance for v1_deepspeed_gather_reduce_scatter_variable_rows.

DeepSpeed ``_AllGatherVariableRows`` backward (pinned 8af59b2e): per-rank
``counts``, each segment zero-padded to ``max_rows`` before the SUM; the official
early return for ``max_rows == 0`` produces an empty tensor without
communicating.  Real Gloo group, 4-tuple return.
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


# (world size, counts per rank, max_rows, non-contiguous gradient)
COMBOS = [
    (1, (3,), 3, False),
    (1, (0,), 0, False),
    (2, (2, 3), 3, True),
    (2, (0, 3), 3, False),
    (2, (0, 0), 0, False),
    (3, (1, 2, 3), 3, True),
]


def input_build(config, rank, dtype=torch.float32):
    world, counts, max_rows, noncontiguous = config
    rows, width = sum(counts), 6
    values = (torch.arange(rows * width, dtype=torch.float32).reshape(rows, width) + rank * 10).to(dtype)
    if noncontiguous:
        storage = torch.empty((rows, width * 2), dtype=dtype)
        storage[:, ::2] = values
        values = storage[:, ::2]
    return {"grad_output": values, "counts": counts, "max_rows": max_rows}


def compute_ref(grad_output, group, counts, max_rows):
    return kernelgenbench.baseline.v1_deepspeed_gather_reduce_scatter_variable_rows(
        grad_output, group, counts, max_rows
    )


def compute_res(grad_output, group, counts, max_rows):
    return kernelgenbench.solution.v1_deepspeed_gather_reduce_scatter_variable_rows(
        grad_output, group, counts, max_rows
    )


def expected_grad(config, rank, dtype):
    world, counts, max_rows, _ = config
    total = sum(input_build(config, other, dtype)["grad_output"] for other in range(world))
    if max_rows == 0:
        return total.new_empty((0, *total.shape[1:]))
    start = sum(counts[:rank])
    return total.narrow(0, start, counts[rank]).clone(memory_format=torch.contiguous_format)


def compare(expected, outputs, config, rank, source):
    world, counts, max_rows, _ = config
    assert isinstance(outputs, tuple) and len(outputs) == 4, (
        f"rank {rank}: _AllGatherVariableRows.backward returns a 4-tuple, got {type(outputs)}"
    )
    grad = outputs[0]
    assert all(value is None for value in outputs[1:])
    assert grad.shape == expected.shape, f"rank {rank}: wrong segment shape {tuple(grad.shape)}"
    assert grad.dtype == expected.dtype
    torch.testing.assert_close(grad, expected, rtol=0, atol=0)
    assert grad.is_contiguous(), f"rank {rank}: the segment must be contiguous"
    assert not torch._C._is_alias_of(grad, source), f"rank {rank}: the segment must be fresh"


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


def _worker(rank, config, dtype, path, outdir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + path, rank=rank,
                            world_size=config[0], timeout=datetime.timedelta(seconds=60))
    try:
        arguments = input_build(config, rank, dtype)
        ref_side = {k: v for k, v in arguments.items()}
        res_side = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in arguments.items()}
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


@label("v1_deepspeed_gather_reduce_scatter_variable_rows")
@parametrize("config", COMBOS)
@parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("config", COMBOS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_accuracy_v1_deepspeed_gather_reduce_scatter_variable_rows(config, dtype):
    with tempfile.TemporaryDirectory(prefix="v1-ds-variable-rs-") as directory:
        mp.start_processes(
            _worker,
            args=(config, dtype, os.path.join(directory, "rdzv"), directory),
            nprocs=config[0],
            join=True,
            start_method="fork",
        )
        timings = [json.load(open(os.path.join(directory, f"{rank}.json"), encoding="utf8"))
                   for rank in range(config[0])]
    ref_time = max(item[0] for item in timings)
    res_time = max(item[1] for item in timings)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
