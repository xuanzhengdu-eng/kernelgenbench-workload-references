"""Accuracy and performance for v1_megatron_cp_to_hp.

Row 549 of the original 723-item table, item OP-c00325593c62-CP2HP.  The Solution
is the real pinned Megatron ``_all_to_all_cp2hp``; the reference states the same
contract with plain tensor ops.

Contract under test
-------------------
Input ``[s, b, h]`` (a per-rank CP shard) becomes ``[s * cp_size, b,
h / cp_size]``: the hidden axis is split into ``cp_size`` chunks, the chunks are
stacked along the token axis, one equal-split all-to-all exchanges them, and the
sequence dimension grows back to ``s * cp_size``.  The input is only read and the
result is fresh.

COMBOS axes: (world size, s, b, h, dtype).  ``h`` is always divisible by the
world size and the per-rank data carries ``rank * 1000`` so a misplaced exchange
is visible.
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
from kernelgenbench.accuracy.v1_synthetic.megatron_cp_test_utils import (
    destroy_group,
    init_group,
    measure,
)


# (world size, s, b, h, dtype)
COMBOS = [
    (2, 4, 2, 8, torch.float32),
    (2, 4, 2, 8, torch.int64),
    (2, 2, 3, 6, torch.float32),
]


def input_build(config, rank=0):
    """Per-rank ``[s, b, h]`` input with ``rank * 1000`` added."""
    world, s, b, h, dtype = config
    count = s * b * h
    values = (torch.arange(count, dtype=torch.int64).reshape(s, b, h) + rank * 1000)
    return {"input_": (values % 100000).to(dtype)}


def oracle(config, rank):
    """Expected ``[s * world, b, h / world]`` output for this rank."""
    world, s, b, h, dtype = config
    per = h // world
    blocks = []
    for source in range(world):
        # Rank ``source`` sends its chunk ``rank`` to this rank, so the output
        # block belonging to ``source`` is that chunk of its flattened input.
        flat = input_build(config, source)["input_"].reshape(s * b, h)
        blocks.append(flat[:, rank * per:(rank + 1) * per])
    # Output block ``k`` is rank ``k``'s chunk ``k``.
    stacked = torch.cat(blocks, dim=0).to(dtype)
    return stacked.reshape(s * world, b, per)


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_megatron_cp_to_hp(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_megatron_cp_to_hp(**inputs)


def compare(ref_out, res_out, config, rank, inputs):
    world, s, b, h, dtype = config
    expected = oracle(config, rank)
    for name, out in (("reference", ref_out), ("solution", res_out)):
        assert isinstance(out, torch.Tensor), name
        assert out.shape == (s * world, b, h // world), (name, out.shape)
        assert out.dtype == dtype
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        assert not torch._C._is_alias_of(out, inputs["input_"]), f"{name} aliases input"


def worker(rank, config, rendezvous, outdir):
    group = init_group(rank, config[0], rendezvous)
    try:
        inputs = input_build(config, rank)
        compare(compute_ref(**inputs, group=group), compute_res(**inputs, group=group),
                config, rank, inputs)
        ref_time = measure(lambda **kw: compute_ref(**kw), {**inputs, "group": group})
        res_time = measure(lambda **kw: compute_res(**kw), {**inputs, "group": group})
        with open(os.path.join(outdir, f"{rank}.json"), "w", encoding="utf8") as handle:
            json.dump([ref_time, res_time], handle)
    finally:
        destroy_group()


@label("v1_megatron_cp_to_hp")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_cp_to_hp(config):
    with tempfile.TemporaryDirectory(prefix="cp-hp-") as directory:
        mp.start_processes(worker,
                           args=(config, os.path.join(directory, "rdzv"), directory),
                           nprocs=config[0], join=True, start_method="fork")
        times = [json.load(open(os.path.join(directory, f"{rank}.json"),
                                encoding="utf8")) for rank in range(config[0])]
    ref_time = max(entry[0] for entry in times)
    res_time = max(entry[1] for entry in times)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
