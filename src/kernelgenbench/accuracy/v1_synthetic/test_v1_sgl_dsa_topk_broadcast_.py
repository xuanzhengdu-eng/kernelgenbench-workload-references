"""Accuracy and performance for v1_sgl_dsa_topk_broadcast_.

Row 551 of the original 723-item table, item OP-2d3cde801fd2.  The Solution is
the real pinned SGLang ``_broadcast_indexer_topk_from_rank0_impl``; the reference
states the same in-place contract.

Contract under test
-------------------
``topk_indices`` (int32 or int64) is overwritten in place with rank 0's values
over the attention-TP group, and the function returns ``None``.  With one rank
no collective runs.  The tensor object/storage is reused (the broadcast writes
into it), and non-source ranks change.

COMBOS axes: (world size, rows, columns, dtype).  Rank ``r`` starts from
``r * 1000 + arange`` so a rank that keeps its own values is detected.
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


# (world size, rows, columns, dtype)
COMBOS = [
    (1, 2, 3, torch.int32),
    (2, 2, 3, torch.int32),
    (2, 2, 3, torch.int64),
    (2, 1, 17, torch.int32),
]


def input_build(config, rank=0):
    world, rows, columns, dtype = config
    values = torch.arange(rows * columns, dtype=torch.int64).reshape(rows, columns)
    return {"topk_indices": (values + rank * 1000).to(dtype)}


def oracle(config):
    return input_build(config, 0)["topk_indices"]


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_sgl_dsa_topk_broadcast_(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_sgl_dsa_topk_broadcast_(**inputs)


def compare(ref_out, res_out, config, ref_inputs, res_inputs, pointers, before):
    world, rows, columns, dtype = config
    expected = oracle(config)
    assert ref_out is None and res_out is None, "both sides return None"
    for name, given, pointer, original in ((("reference"), ref_inputs["topk_indices"],
                                            pointers[0], before[0]),
                                           ("solution", res_inputs["topk_indices"],
                                            pointers[1], before[1])):
        assert given.dtype == dtype
        assert given.data_ptr() == pointer, f"{name}: buffer was replaced"
        if world == 1:
            torch.testing.assert_close(given, original, rtol=0, atol=0)
        else:
            torch.testing.assert_close(given, expected, rtol=0, atol=0)


def worker(rank, config, rendezvous, outdir):
    group = init_group(rank, config[0], rendezvous)
    try:
        # Each side gets its own buffer: the operation writes in place, so a
        # shared buffer would hide a wrong source rank.
        ref_inputs = input_build(config, rank)
        res_inputs = input_build(config, rank)
        pointers = (ref_inputs["topk_indices"].data_ptr(),
                    res_inputs["topk_indices"].data_ptr())
        before = (ref_inputs["topk_indices"].clone(), res_inputs["topk_indices"].clone())
        ref_out = compute_ref(topk_indices=ref_inputs["topk_indices"], group=group)
        res_out = compute_res(topk_indices=res_inputs["topk_indices"], group=group)
        compare(ref_out, res_out, config, ref_inputs, res_inputs, pointers, before)
        ref_time = measure(lambda **kw: compute_ref(**kw),
                           {"topk_indices": input_build(config, rank)["topk_indices"],
                            "group": group})
        res_time = measure(lambda **kw: compute_res(**kw),
                           {"topk_indices": input_build(config, rank)["topk_indices"],
                            "group": group})
        with open(os.path.join(outdir, f"{rank}.json"), "w", encoding="utf8") as handle:
            json.dump([ref_time, res_time], handle)
    finally:
        destroy_group()


@label("v1_sgl_dsa_topk_broadcast_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_dsa_topk_broadcast_(config):
    with tempfile.TemporaryDirectory(prefix="v1-dsa-topk-") as directory:
        mp.start_processes(worker,
                           args=(config, os.path.join(directory, "rdzv"), directory),
                           nprocs=config[0], join=True, start_method="fork")
        times = [json.load(open(os.path.join(directory, f"{rank}.json"),
                                encoding="utf8")) for rank in range(config[0])]
    ref_time = max(entry[0] for entry in times)
    res_time = max(entry[1] for entry in times)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
