"""Accuracy and performance for v1_sgl_cp_zigzag_gather.

Row 546 of the original 723-item table, item OP-d2f517a79190-SGL, zigzag
strategy.  The Solution is the real pinned ``ZigzagCPStrategy.gather_hidden_states``
(which first calls ``_all_gather_reorganized``); the reference repeats the same
steps with plain tensor ops.

Contract under test
-------------------
Each rank holds ``[physical, width]`` whose first ``logical_lengths[rank]`` rows
are real.  The function trims to that length, zero-pads to
``max(physical_lengths)``, all-gathers, trims every gathered block back to its
logical length, concatenates them, then splits by ``split_lengths`` and reorders
by ``reverse_indices`` (the inverse zigzag permutation).

COMBOS axes: (cp_size, total tokens, feature width, split lengths, reverse
index).  ``split_lengths`` always sums to ``total`` and ``reverse_indices`` is a
permutation of ``range(len(split_lengths))``.
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
from kernelgenbench.accuracy.v1_synthetic.sgl_cp_test_utils import (
    destroy_group,
    init_group,
    install_cp_group,
    measure,
)


# (cp_size, total tokens, width, split lengths, reverse index)
COMBOS = [
    (2, 6, 8, (3, 3), (0, 1)),
    (2, 8, 4, (2, 2, 2, 2), (0, 2, 1, 3)),
    (2, 7, 5, (2, 2, 2, 1), (0, 3, 2, 1)),
    (4, 8, 6, (2, 2, 2, 2), (2, 0, 3, 1)),
]


def logical_lengths(config):
    cp_size, total = config[0], config[1]
    return [(total + cp_size - 1 - rank) // cp_size for rank in range(cp_size)]


def input_build(config, rank=0):
    """Per-rank input: ``[physical, width]`` float32 with zero padding rows."""
    cp_size, total, width, _, _ = config
    logical = logical_lengths(config)
    physical = max(logical)
    generator = torch.Generator(device="cpu").manual_seed(20260946 + rank)
    rows = torch.arange(physical * width, dtype=torch.float32).reshape(physical, width)
    rows = (rows + rank * 1000.0).remainder(997)
    if logical[rank] < physical:
        rows[logical[rank]:] = 0.0
    return {
        "x": rows.contiguous(),
        "logical_lengths": logical,
        "physical_lengths": [physical] * cp_size,
        "total_tokens": total,
        "split_lengths": list(config[3]),
        "reverse_indices": list(config[4]),
    }


def oracle(config):
    """Expected global zigzag order rebuilt from the per-rank inputs."""
    cp_size, total, width, split_lengths, reverse = config
    logical = logical_lengths(config)
    blocks = [input_build(config, rank)["x"][: logical[rank]] for rank in range(cp_size)]
    joined = torch.cat(blocks, dim=0)
    pieces = torch.split(joined, list(split_lengths), dim=0)
    return torch.cat([pieces[each] for each in reverse], dim=0).contiguous()


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_sgl_cp_zigzag_gather(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_sgl_cp_zigzag_gather(**inputs)


def compare(ref_out, res_out, config):
    expected = oracle(config)
    for name, out in (("reference", ref_out), ("solution", res_out)):
        assert isinstance(out, torch.Tensor), name
        assert out.shape == expected.shape, (name, out.shape, expected.shape)
        assert out.dtype == torch.float32 and out.is_contiguous()
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


def worker(rank, config, rendezvous, outdir):
    group = init_group(rank, config[0], rendezvous)
    try:
        inputs = input_build(config, rank)
        with install_cp_group(config[0], rank, group):
            ref_out = compute_ref(**inputs, group=group)
            res_out = compute_res(**inputs, group=group)
            compare(ref_out, res_out, config)
            ref_time = measure(lambda **kw: compute_ref(**kw), {**inputs, "group": group},
                               group)
            res_time = measure(lambda **kw: compute_res(**kw), {**inputs, "group": group},
                               group)
        with open(os.path.join(outdir, f"{rank}.json"), "w", encoding="utf8") as handle:
            json.dump([ref_time, res_time], handle)
    finally:
        destroy_group()


@label("v1_sgl_cp_zigzag_gather")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_cp_zigzag_gather(config):
    with tempfile.TemporaryDirectory(prefix="v1-cp-zigzag-") as directory:
        mp.start_processes(worker,
                           args=(config, os.path.join(directory, "rdzv"), directory),
                           nprocs=config[0], join=True, start_method="fork")
        times = [json.load(open(os.path.join(directory, f"{rank}.json"),
                                encoding="utf8")) for rank in range(config[0])]
    ref_time = max(entry[0] for entry in times)
    res_time = max(entry[1] for entry in times)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
