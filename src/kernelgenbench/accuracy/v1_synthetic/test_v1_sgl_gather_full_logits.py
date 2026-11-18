"""Accuracy and timing for v1_sgl_gather_full_logits.

Two Gloo ranks each hold a (rows, local_vocab) logits shard.  The real
sharded-verify entry all-gathers flattened shards in rank order, reshapes to
(rows, world * local_vocab), crops and casts to FP32.  Empty chunks, crop
smaller/equal to the gathered width and world size 1/2 are covered.
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
from kernelgenbench.accuracy.v1_synthetic.rows642_651_test_utils import (
    clone_tree,
    measure_ms,
)
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (rows, local_vocab, vocab_size, world, dtype)
COMBOS = [
    (2, 5, 8, 2, torch.float32),
    (3, 4, 8, 2, torch.float32),
    (4, 6, 10, 2, torch.float32),
    (2, 3, 6, 2, torch.float32),
    (2, 5, 8, 1, torch.float16),
    (3, 4, 7, 1, torch.bfloat16),
]


class _Group:
    """Minimal tp_group view exposing the two attributes upstream reads."""

    def __init__(self, process_group, world_size):
        self.process_group = process_group
        self.world_size = world_size

    def all_gather_into_tensor(self, output, input_):
        dist.all_gather_into_tensor(
            output, input_, group=self.process_group
        )


def input_build(config):
    rows, local_vocab, vocab_size, world, dtype = config
    generator = torch.Generator().manual_seed(645)
    return {
        "logits": torch.randn(
            (rows, local_vocab), dtype=dtype, generator=generator
        ),
        "vocab_size": vocab_size,
        "world": world,
        "dtype": dtype,
    }


def _rank_inputs(base, rank, world):
    logits = base["logits"] + rank * 100.0
    rank_logits = [base["logits"] + r * 100.0 for r in range(world)]
    return logits, rank_logits


def compute_ref(logits, vocab_size, rank_logits, group=None):
    return kernelgenbench.baseline.v1_sgl_gather_full_logits(
        logits, vocab_size, rank_logits
    )


def compute_res(logits, vocab_size, rank_logits=None, group=None):
    return kernelgenbench.solution.v1_sgl_gather_full_logits(
        logits, vocab_size, rank_logits, group
    )


def compare(ref_out, res_out, vocab_size, world, local_vocab):
    width = min(vocab_size, world * local_vocab)
    assert isinstance(res_out, torch.Tensor)
    assert res_out.dtype == ref_out.dtype == torch.float32
    assert res_out.shape == ref_out.shape
    assert res_out.shape[-1] == width
    torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0)


def _worker(rank, config, init_file, out_dir):
    rows, local_vocab, vocab_size, world, dtype = config
    dist.init_process_group(
        "gloo",
        init_method="file://" + init_file,
        rank=rank,
        world_size=world,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        base = input_build(config)
        logits, rank_logits = _rank_inputs(base, rank, world)
        group = _Group(dist.group.WORLD, world)
        inputs = {
            "logits": logits,
            "vocab_size": vocab_size,
            "rank_logits": rank_logits,
            "group": group,
        }
        before = clone_tree(logits)
        ref_out = compute_ref(**inputs)
        res_out = compute_res(**inputs)
        compare(ref_out, res_out, vocab_size, world, local_vocab)
        torch.testing.assert_close(logits, before, rtol=0, atol=0)

        ref_ms = compute_ref_performance(**inputs)
        res_ms = compute_res_performance(**inputs)
        with open(os.path.join(out_dir, f"{rank}.json"), "w") as handle:
            json.dump({"ref_ms": ref_ms, "res_ms": res_ms}, handle)
    finally:
        dist.destroy_process_group()


def compute_ref_performance(**inputs):
    return measure_ms(
        compute_ref,
        inputs,
        torch.device("cpu"),
        repetitions=10,
        keep=("group",),
    )


def compute_res_performance(**inputs):
    return measure_ms(
        compute_res,
        inputs,
        torch.device("cpu"),
        repetitions=10,
        keep=("group",),
    )


@label("v1_sgl_gather_full_logits")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_gather_full_logits(config):
    rows, local_vocab, vocab_size, world, dtype = config
    with tempfile.TemporaryDirectory(prefix="v1-gather-logits-") as directory:
        init_file = os.path.join(directory, "rdzv")
        mp.start_processes(
            _worker,
            args=(config, init_file, directory),
            nprocs=world,
            join=True,
            start_method="fork",
        )
        times = [
            json.load(open(os.path.join(directory, f"{rank}.json")))
            for rank in range(world)
        ]
    ref_time = max(item["ref_ms"] for item in times)
    res_time = max(item["res_ms"] for item in times)
    return CustomBenchmarkResult(
        ref_time=ref_time,
        res_time=res_time,
        speedup=ref_time / res_time if res_time else float("inf"),
    )
