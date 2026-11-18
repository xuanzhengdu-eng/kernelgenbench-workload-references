"""Accuracy and performance for v1_sgl_tp_residual_gather_.

Item: row 317 (OP-508adcfe879b).  SGLang attention-TP output boundary: the
incoming gradient rows are added into the residual in place (``PlainAdd``), then
the local rows are all-gathered in rank order; the second return value is
``None``.  The hidden tensor keeps its storage and is updated; the residual
input is not touched.  ``group`` is the attention-TP process group.
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
from sandbox.verifier.test_parametrize import label, parametrize
from sandbox.utils.accuracy_utils import CustomBenchmarkResult

# (world size, token rows per rank, hidden width, dtype)
CASES = [
    (1, 3, 5, torch.float32),
    (2, 4, 7, torch.float32),
    (3, 2, 4, torch.bfloat16),
    (2, 0, 5, torch.float32),
    (2, 5, 3, torch.float16),
]
COMBOS = [(case, rank) for case in CASES for rank in range(case[0])]


def input_build(config):
    case, rank = config
    world, rows, width, dtype = case
    hidden = (torch.arange(rows * width, dtype=torch.float32).reshape(rows, width)
              / 8 + rank).to(dtype)
    residual = torch.full((rows, width), rank + 1, dtype=dtype)
    return {"hidden_states": hidden, "residual": residual}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_sgl_tp_residual_gather_(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_sgl_tp_residual_gather_(**inputs)


def compare(ref_out, res_out):
    assert isinstance(res_out, tuple) and len(res_out) == 2, type(res_out)
    assert ref_out[1] is None and res_out[1] is None
    assert res_out[0].shape == ref_out[0].shape
    assert res_out[0].dtype == ref_out[0].dtype
    torch.testing.assert_close(res_out[0], ref_out[0], rtol=0, atol=0)


def expected_rows(case, rank):
    world, rows, width, dtype = case
    local = input_build((case, rank))["hidden_states"] + \
        input_build((case, rank))["residual"]
    return local


def worker(rank, case, path, outdir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + path, rank=rank,
                            world_size=case[0],
                            timeout=datetime.timedelta(seconds=120))
    try:
        world, rows, width, dtype = case
        inputs = input_build((case, rank))
        ref_inputs = {name: value.clone() for name, value in inputs.items()}
        res_inputs = {name: value.clone() for name, value in inputs.items()}
        original_pointer = res_inputs["hidden_states"].data_ptr()
        ref_out = compute_ref(**ref_inputs, group=dist.group.WORLD)
        res_out = compute_res(**res_inputs, group=dist.group.WORLD)
        compare(ref_out, res_out)

        assert tuple(res_out[0].shape) == (world * rows, width)
        # Rank order: every rank's updated rows, in rank order.
        for source in range(world):
            torch.testing.assert_close(
                res_out[0][source * rows:(source + 1) * rows],
                expected_rows(case, source), rtol=0, atol=0)
        # In-place: hidden_states keeps its storage and now holds this rank's sum;
        # the gathered result is a fresh tensor built by cat.
        assert res_inputs["hidden_states"].data_ptr() == original_pointer
        assert not torch._C._is_alias_of(res_out[0], res_inputs["hidden_states"])
        torch.testing.assert_close(res_inputs["hidden_states"],
                                   expected_rows(case, rank), rtol=0, atol=0)
        # The residual input is only read.
        for side in (ref_inputs, res_inputs):
            torch.testing.assert_close(side["residual"], inputs["residual"],
                                       rtol=0, atol=0)

        def bench(fn, made):
            dist.barrier()
            begin = time.perf_counter()
            for _ in range(5):
                fn(**made, group=dist.group.WORLD)
            dist.barrier()
            return (time.perf_counter() - begin) * 1000 / 5

        ref_time = bench(compute_ref, ref_inputs)
        res_time = bench(compute_res, res_inputs)
        with open(os.path.join(outdir, str(rank) + ".json"), "w") as handle:
            json.dump([ref_time, res_time], handle)
    finally:
        dist.destroy_process_group()


@label("v1_sgl_tp_residual_gather_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_tp_residual_gather_(config):
    case, rank = config
    with tempfile.TemporaryDirectory(prefix="v1-tp-residual-") as directory:
        mp.start_processes(
            worker, args=(case, os.path.join(directory, "rdzv"), directory),
            nprocs=case[0], join=True, start_method="fork")
        times = [json.load(open(os.path.join(directory, str(r) + ".json")))
                 for r in range(case[0])]
    ref_time = max(entry[0] for entry in times)
    res_time = max(entry[1] for entry in times)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
