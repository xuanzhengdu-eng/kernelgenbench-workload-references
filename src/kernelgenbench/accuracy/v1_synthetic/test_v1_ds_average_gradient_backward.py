"""DeepSpeed _AverageGradientOverTP.backward: mean of the TP-folded gradient.

Official source: deepspeed/moe/ep_tp_dispatch.py, ``_AverageGradientOverTP`` --
forward is the identity, backward clones the incoming gradient in contiguous
format, all-reduces it over the TP group and divides by the world size, so every
peer holds the true average.  The reference keeps the original contract: a 2-tuple
``(mean, None)`` with a fresh contiguous tensor and an untouched input.
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

# world size, rows, columns, non-contiguous input
COMBOS = [(world, rows, cols, strided)
          for world in (1, 2)
          for rows, cols in ((2, 3), (5, 7))
          for strided in (False, True)]


def input_build(config, rank, dtype=torch.float32):
    _, rows, cols, strided = config
    grad = torch.arange(rows * cols, dtype=dtype).reshape(rows, cols) + rank * 10
    if strided:
        wide = torch.full((rows, cols * 2), -99, dtype=dtype)
        wide[:, ::2] = grad
        grad = wide[:, ::2]
    return {"grad_output": grad}


def compute_ref(grad_output, group):
    return kernelgenbench.baseline.v1_ds_average_gradient_backward(grad_output, group)


def compute_res(grad_output, group):
    return kernelgenbench.solution.v1_ds_average_gradient_backward(grad_output, group)


def compare(config, rank, ref_out, res_out, original, dtype):
    world, rows, cols, strided = config
    assert isinstance(ref_out, tuple) and isinstance(res_out, tuple)
    assert len(ref_out) == len(res_out) == 2
    assert ref_out[1] is None and res_out[1] is None, "group argument has no gradient"
    expected = torch.zeros(rows, cols, dtype=dtype)
    for peer in range(world):
        expected += input_build(config, peer, dtype)["grad_output"]
    expected = expected / world
    torch.testing.assert_close(ref_out[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(res_out[0], expected, rtol=0, atol=0)
    assert res_out[0].shape == (rows, cols) and res_out[0].dtype == dtype
    assert res_out[0].is_contiguous(), "average must be contiguous"
    if res_out[0].numel():
        assert res_out[0].data_ptr() != original.data_ptr(), "average must be a fresh tensor"
    assert torch.equal(original, input_build(config, rank, dtype)["grad_output"]), "input mutated"


def _worker(rank, config, path, outdir, dtype):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + path, rank=rank,
                            world_size=config[0], timeout=datetime.timedelta(seconds=60))
    try:
        payload = input_build(config, rank, dtype)
        grad = payload["grad_output"]
        before = grad.clone()
        group = dist.group.WORLD
        ref_out = compute_ref(**payload, group=group)
        res_out = compute_res(**payload, group=group)
        compare(config, rank, ref_out, res_out, grad, grad.dtype)
        assert torch.equal(grad, before)
        timings = []
        for fn in (compute_ref, compute_res):
            dist.barrier()
            samples = []
            for _ in range(3):
                start = time.perf_counter()
                fn(**payload, group=group)
                dist.barrier()
                samples.append((time.perf_counter() - start) * 1000)
            timings.append(min(samples))
        with open(os.path.join(outdir, f"{rank}.json"), "w", encoding="utf8") as handle:
            json.dump(timings, handle)
    finally:
        dist.destroy_process_group()


@label("v1_ds_average_gradient_backward")
@parametrize("config", COMBOS)
@parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("config,dtype", [(config, dtype) for config in COMBOS
                                          for dtype in (torch.float32, torch.float64)])
def test_accuracy_v1_ds_average_gradient_backward(config, dtype):
    world = config[0]
    with tempfile.TemporaryDirectory(prefix="v1-average-gradient-") as outdir:
        mp.start_processes(_worker, args=(config, os.path.join(outdir, "rdzv"), outdir, dtype),
                           nprocs=world, join=True, start_method="fork")
        times = [json.load(open(os.path.join(outdir, f"{rank}.json"), encoding="utf8"))
                 for rank in range(world)]
    ref_time, res_time = max(item[0] for item in times), max(item[1] for item in times)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
