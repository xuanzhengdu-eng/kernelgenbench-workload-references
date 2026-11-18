"""SGLang sharded-target greedy token selection over a real 2-rank group.

One Item, one reference/test.  The test builds each rank's vocabulary shard on
the GPU, runs the reference and the official Triton key exchange, and checks the
int64 token ids plus the NaN / tie / -inf behaviour the contract promises.
"""
import datetime
import json
import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

# world size, dtype, rows
COMBOS = [
    (1, torch.float32, 4),
    (2, torch.float32, 4),
    (2, torch.bfloat16, 5),
]
WIDTH = 8


def input_build(config, rank):
    world, dtype, rows = config
    start = rank * WIDTH
    valid = WIDTH if rank == 0 else 5
    vocab = (world - 1) * WIDTH + (5 if world > 1 else 8)
    x = torch.arange(rows * WIDTH, dtype=torch.float32).reshape(rows, WIDTH).remainder(5) - 2
    x[0, :] = 0
    x[0, 0] = -0.0
    x[1, 0] = float("nan")
    x[1, 1] = 3
    x[2, :] = -float("inf")
    x[3, :] = -3
    x[3, 0] = float("inf")
    if rank == 1:
        x[1, 2] = 4
    return {
        "logits": x.to(dtype).cuda(),
        "shard_start": start,
        "valid": valid,
        "vocab_size": vocab,
    }


def compute_ref(**kwargs):
    return kernelgenbench.baseline.v1_sgl_sharded_target_tokens(**kwargs)


def compute_res(**kwargs):
    return kernelgenbench.solution.v1_sgl_sharded_target_tokens(**kwargs)


def compare(ref_out, res_out, rows):
    """Contract: int64 ``[rows]`` greedy ids; ties pick the *smaller* token id."""
    assert isinstance(res_out, torch.Tensor) and isinstance(ref_out, torch.Tensor)
    assert ref_out.dtype == res_out.dtype == torch.int64
    assert ref_out.shape == res_out.shape == (rows,)
    assert torch.equal(res_out, ref_out), (res_out, ref_out)
    # Row 0 is all zeros: the smallest token id 0 wins on every rank.
    assert res_out[0].item() == 0
    # Row 2 is all -inf: the tie still resolves to the smallest id.
    assert res_out[2].item() == 0
    # Row 3 holds +inf; a finite maximum is the only winner.
    assert res_out[3].item() == 0
    assert torch.all(res_out >= 0) and torch.all(res_out < 0xFFFFFFFF)


def _negative(logits, shard_start, valid, vocab_size, group):
    """Break the tie rule: report the largest instead of the smallest token id."""
    best = kernelgenbench.baseline.v1_sgl_sharded_target_tokens(
        logits, shard_start, valid, vocab_size, group
    )
    return vocab_size - 1 - best


def _worker(rank, config, init_file, out_dir):
    import os

    import solutions_sglang
    from types import SimpleNamespace

    entries = dict(solutions_sglang.ENTRIES)
    if os.environ.get("KGB_ROW719_NEGATIVE") == "1":
        entries["v1_sgl_sharded_target_tokens"] = _negative
    kernelgenbench.triton = SimpleNamespace(**entries)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    dist.init_process_group(
        "gloo", init_method="file://" + init_file, rank=rank,
        world_size=config[0], timeout=datetime.timedelta(seconds=60),
    )
    try:
        data = input_build(config, rank)
        logits_before = data["logits"].clone()
        expected = compute_ref(**data, group=dist.group.WORLD)
        res_inputs = input_build(config, rank)
        actual = compute_res(**res_inputs, group=dist.group.WORLD)
        compare(expected, actual, config[2])
        # The reference must not mutate its logits (NaN stays NaN).
        torch.testing.assert_close(
            data["logits"], logits_before, rtol=0, atol=0, equal_nan=True
        )
        torch.testing.assert_close(
            res_inputs["logits"], logits_before, rtol=0, atol=0, equal_nan=True
        )

        def bench(fn):
            return compute_res_performance(config, rank, fn)

        ref_time, res_time = bench(compute_ref), bench(compute_res)
        with open(os.path.join(out_dir, f"{rank}.json"), "w") as handle:
            json.dump([ref_time, res_time], handle)
    finally:
        dist.destroy_process_group()


def _timed(fn, config, rank, warmup, repeats):
    for _ in range(warmup):
        fn(**input_build(config, rank), group=dist.group.WORLD)
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn(**input_build(config, rank), group=dist.group.WORLD)
    end.record()
    end.synchronize()
    dist.barrier()
    return start.elapsed_time(end) / repeats


def compute_ref_performance(config, rank, fn=None):
    """Median-free mean of ten round trips of this rank's reference entry."""
    return _timed(fn or compute_ref, config, rank, warmup=2, repeats=10)


def compute_res_performance(config, rank, fn=None):
    """Same timing loop for the bound solution entry."""
    return _timed(fn or compute_res, config, rank, warmup=2, repeats=10)


@label("v1_sgl_sharded_target_tokens")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_sharded_target_tokens(config):
    with tempfile.TemporaryDirectory(prefix="v1-sharded-target-") as out_dir:
        mp.start_processes(
            _worker,
            args=(config, os.path.join(out_dir, "rdzv"), out_dir),
            nprocs=config[0],
            join=True,
            start_method="spawn",
        )
        times = [json.load(open(os.path.join(out_dir, f"{rank}.json"))) for rank in range(config[0])]
    ref_time = max(entry[0] for entry in times)
    res_time = max(entry[1] for entry in times)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time)
