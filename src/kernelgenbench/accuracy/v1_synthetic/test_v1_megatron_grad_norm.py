"""Accuracy and performance for v1_megatron_grad_norm.

Operator: megatron.core.optimizer.clip_grads.get_grad_norm_fp32 with the official
local multi-tensor fallback (no Transformer Engine / Apex).
Official source (pinned commit):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/optimizer/clip_grads.py#L55-L142
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/utils.py#L1178-L1186

Item: v1_megatron_grad_norm (723-row 169). ``compute_ref`` loads this file's
reference from ``kernelgenbench.baseline``; ``compute_res`` calls the real official
``get_grad_norm_fp32`` bound as ``kernelgenbench.solution``. The old reference used
a fork link and CPU/Gloo tensors; the official entry exists at the pinned commit,
so this item now runs the official operator with real NCCL collectives.

Every case starts 1 or 2 worker processes with a file-store process group. Each
rank holds different gradient tensors, so only a real all-reduce can produce the
independent scalar computed from all ranks' values. Cases cover p=1, 2, 3, inf,
an empty list on the L2 branch, and 1-rank/2-rank worlds. The assertion compares
both implementations against the closed-form value and checks the return is a
float.

Multi-rank note: this batch has a single visible H800 (GPU 3), and NCCL rejects two
ranks that share one device ("duplicate GPU", invalid usage), so the 2-rank cases
use the Gloo backend with CPU tensors on the branches the official fallback keeps
on CPU (p=1 and p=3). The p=2 and inf branches force a CUDA fp32 scalar upstream,
so they are exercised with a real one-rank NCCL group; their cross-rank reduction
uses the same all_reduce call but cannot be run with two ranks here.
"""
import datetime
import json
import math
import os
import tempfile
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows162_171_checks as checks


# (world size, backend, tensor device, norm type, empty gradient list)
COMBOS = [
    (1, "nccl", "cuda", 2.0, False),
    (1, "nccl", "cuda", float('inf'), False),
    (1, "nccl", "cuda", 2.0, True),
    (2, "gloo", "cpu", 1.0, False),
    (2, "gloo", "cpu", 3.0, False),
]


def _rank_grads(case, rank):
    _, _, _, norm_type, empty = case
    if empty:
        return []
    return [
        torch.tensor([rank + 1.0, -2.0, 0.0], dtype=torch.float32),
        torch.tensor([3.0, rank + 2.0], dtype=torch.float32),
    ]


def input_build(case):
    return {"case": case}


def _independent(case, world):
    _, _, _, p, empty = case
    if empty:
        return 0.0
    values = [float(x) for r in range(world) for g in _rank_grads(case, r)
              for x in g.tolist()]
    if p == float('inf'):
        return max(abs(x) for x in values)
    return sum(abs(x) ** p for x in values) ** (1.0 / p)


def compute_ref(grads_for_norm, norm_type, group):
    return kernelgenbench.baseline.v1_megatron_grad_norm(
        grads_for_norm=grads_for_norm, norm_type=norm_type,
        grad_stats_parallel_group=group)


def compute_res(grads_for_norm, norm_type, group):
    return kernelgenbench.solution.v1_megatron_grad_norm(
        grads_for_norm=grads_for_norm, norm_type=norm_type,
        grad_stats_parallel_group=group)


NAME = "v1_megatron_grad_norm"


def _install_solution(solution):
    """Bind the candidate inside a spawned rank.

    ``spawn`` starts a fresh interpreter, so the parent's ``kernelgenbench.solution``
    binding is not inherited; the parent passes the same callable it resolved from
    ``kernelgenbench.solution`` and each rank re-installs it under the public
    ``kernelgenbench.triton`` namespace the test entry point reads.
    """
    from types import SimpleNamespace
    namespace = getattr(kernelgenbench, "triton", None)
    if namespace is None:
        namespace = SimpleNamespace()
        kernelgenbench.triton = namespace
    setattr(namespace, NAME, solution)


def _worker(rank, case, store_path, out_dir, solution):
    world, backend, tensor_device, norm_type, empty = case
    torch.set_num_threads(1)
    if backend == "nccl":
        torch.cuda.set_device(0)
        _install_solution(solution)
    dist.init_process_group(backend, init_method="file://" + store_path, rank=rank,
                            world_size=world, timeout=datetime.timedelta(seconds=60))
    try:
        grads = [g.to(tensor_device) for g in _rank_grads(case, rank)]
        group = dist.group.WORLD
        ref_value = compute_ref(grads, norm_type, group)
        res_value = compute_res(grads, norm_type, group)
        expected = _independent(case, world)
        assert isinstance(ref_value, float) and isinstance(res_value, float), \
            "both implementations must return a Python float"
        assert math.isclose(ref_value, res_value, rel_tol=1e-6, abs_tol=1e-6), \
            f"reference {ref_value} != candidate {res_value}"
        assert math.isclose(res_value, expected, rel_tol=1e-6, abs_tol=1e-6), \
            f"candidate {res_value} != independent {expected}"

        def bench(fn):
            dist.barrier()
            started = time.perf_counter()
            for _ in range(3):
                fn(grads, norm_type, group)
            dist.barrier()
            return (time.perf_counter() - started) * 1000 / 3

        ref_time = bench(compute_ref)
        res_time = bench(compute_res)
        with open(os.path.join(out_dir, f"{rank}.json"), "w", encoding="utf8") as handle:
            json.dump([ref_time, res_time], handle)
    finally:
        dist.destroy_process_group()


@label("v1_megatron_grad_norm")
@parametrize("case", COMBOS)
@pytest.mark.parametrize("case", COMBOS)
def test_accuracy_v1_megatron_grad_norm(case):
    world = case[0]
    solution = kernelgenbench.solution.v1_megatron_grad_norm
    with tempfile.TemporaryDirectory(prefix="v1-grad-norm-") as directory:
        store = os.path.join(directory, "store")
        # Gloo/CPU ranks do not initialize CUDA, so fork is safe and keeps the
        # parent's solution binding; NCCL needs a fresh interpreter for CUDA.
        method = "spawn" if case[1] == "nccl" else "fork"
        mp.start_processes(_worker, args=(case, store, directory, solution),
                           nprocs=world, join=True, start_method=method)
        times = [json.load(open(os.path.join(directory, f"{rank}.json"),
                                encoding="utf8")) for rank in range(world)]
    ref_time = max(pair[0] for pair in times)
    res_time = max(pair[1] for pair in times)
    return checks.timings(ref_time, res_time)
