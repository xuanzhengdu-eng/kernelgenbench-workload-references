"""Accuracy and performance for v1_sgl_reduce_sharded_token_keys.

Official operator: SGLang ``reduce_sharded_token_keys`` (sharded_verify.py) on a
real two-rank NCCL group. Keys are int64 ``score << 32 | (0xffffffff - token)``;
the largest key wins across ranks and splits, ties break toward the smaller
token.

Rows stay above the upstream 1024-row multimem ceiling so the documented
all-gather fallback runs: on one shared H800 the LL/multimem symmetric-memory
setup cannot be used by two ranks, and the official module logs that fallback.
"""
import datetime
import importlib
import importlib.util
import json
import os
import sys
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic import upstream_op_test_utils as util
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

NAME = "v1_sgl_reduce_sharded_token_keys"

# Two ranks share the single visible H800, which NCCL requires an opt-in for.
os.environ.setdefault("NCCL_MULTI_RANK_GPU_ENABLE", "1")

# world size, splits, rows, equal-score tie case
COMBOS = [
    (2, 1, 1025, False),
    (2, 2, 1026, False),
    (2, 4, 1032, False),
    (2, 2, 1025, True),
]


def keys_for(world, splits, rows, tie):
    """Keys of every rank, shaped [world, rows, splits] (or [world, rows])."""
    rank_ids = torch.arange(world, dtype=torch.int64).reshape(world, 1, 1)
    row_ids = torch.arange(rows, dtype=torch.int64).reshape(1, rows, 1)
    part_ids = torch.arange(splits, dtype=torch.int64).reshape(1, 1, splits)
    score = 1000 + row_ids * 7 + rank_ids * 3 + part_ids
    token = 5000 + row_ids * 5 + rank_ids + part_ids * 2
    if tie:
        # Same score on both ranks, different tokens: the smaller token must win.
        score = score * 0 + 1000
    keys = (score << 32) | (0xFFFFFFFF - token)
    return keys[:, :, 0] if splits == 1 else keys


def expected_tokens(world, splits, rows, tie):
    keys = keys_for(world, splits, rows, tie)
    best = keys[0].clone()
    for rank in range(1, world):
        best = torch.maximum(best, keys[rank])
    if best.dim() == 2:
        best = best.amax(dim=1)
    return 0xFFFFFFFF - (best & 0xFFFFFFFF)


def input_build(config, rank):
    world, splits, rows, tie = config
    keys = keys_for(world, splits, rows, tie)[rank]
    return {"keys": keys.contiguous(), "splits": splits}


def compute_ref(keys, group, splits):
    return kernelgenbench.baseline.v1_sgl_reduce_sharded_token_keys(keys, group, splits)


def compute_res(keys, group, splits):
    return kernelgenbench.solution.v1_sgl_reduce_sharded_token_keys(keys, group, splits)


def compare(ref_out, res_out, config):
    assert isinstance(res_out, torch.Tensor)
    world, splits, rows, tie = config
    assert res_out.shape == ref_out.shape == (rows,)
    assert res_out.dtype == ref_out.dtype == torch.int64
    torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0)


def _solution_spec():
    """File path and name of the bound Solution, for the spawned ranks."""
    entry = getattr(kernelgenbench.solution, NAME)
    module = sys.modules[entry.__module__]
    path = getattr(module, "__file__", None)
    assert path and "<locals>" not in entry.__qualname__, (
        "spawned ranks need a module-level solution on disk")
    return path, entry.__qualname__


def _bind_solution_in_child(spec):
    path, qualname = spec
    # The binding may live in the batch's verify directory, which is not on the
    # spawned interpreter's sys.path; load it from its file instead.
    loader = importlib.util.spec_from_file_location("kgb_row_solution", path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    entry = getattr(module, qualname)
    namespace = getattr(kernelgenbench, "triton", None)
    if namespace is None:
        namespace = SimpleNamespace()
        kernelgenbench.triton = namespace
    setattr(namespace, NAME, entry)


def _timed(fn, inputs, group):
    """One timed call with the ranks parked on either side."""
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn(**util.fresh(inputs, kernelgenbench.device), group=group)
    end.record()
    dist.barrier()
    end.synchronize()
    return start.elapsed_time(end)


def _worker(rank, index, spec, init_file, out_dir):
    _bind_solution_in_child(spec)
    torch.cuda.set_device(0)
    config = COMBOS[index]
    dist.init_process_group("nccl", init_method="file://" + init_file, rank=rank,
                            world_size=config[0],
                            timeout=datetime.timedelta(seconds=120))
    try:
        group = dist.group.WORLD
        inputs = input_build(config, rank)
        ref_inputs = util.fresh(inputs, kernelgenbench.device)
        res_inputs = util.fresh(inputs, kernelgenbench.device)
        ref_out = compute_ref(**ref_inputs, group=group)
        res_out = compute_res(**res_inputs, group=group)
        compare(ref_out, res_out, config)
        torch.testing.assert_close(res_out, expected_tokens(*config).to(ref_out.device),
                                   rtol=0, atol=0)
        torch.testing.assert_close(res_inputs["keys"].cpu(), inputs["keys"], rtol=0, atol=0)
        if rank == 0:
            print(f"OK reduce_sharded config{index} splits={config[1]} rows={config[2]} "
                  f"tokens={res_out[:3].tolist()}", flush=True)
        for _ in range(3):
            _timed(compute_ref, inputs, group)
            _timed(compute_res, inputs, group)
        samples = [(_timed(compute_ref, inputs, group),
                    _timed(compute_res, inputs, group)) for _ in range(5)]
        with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as handle:
            json.dump({"ref": sorted(s[0] for s in samples)[2],
                       "res": sorted(s[1] for s in samples)[2]}, handle)
    finally:
        dist.destroy_process_group()


@label("v1_sgl_reduce_sharded_token_keys")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_reduce_sharded_token_keys(config):
    world = config[0]
    index = COMBOS.index(config)
    with tempfile.TemporaryDirectory(prefix="sharded-keys-") as out_dir:
        mp.start_processes(_worker, args=(index, _solution_spec(),
                                         os.path.join(out_dir, "rdzv"), out_dir),
                           nprocs=world, join=True, start_method="spawn")
        timings = [json.load(open(os.path.join(out_dir, f"rank{rank}.json")))
                   for rank in range(world)]
    ref_time = max(entry["ref"] for entry in timings)
    res_time = max(entry["res"] for entry in timings)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
