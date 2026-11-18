"""Accuracy and performance for v1_sgl_dcp_extend_kv_gather_.

DCP ranks hold interleaved prefix rows.  The official entry all-gathers the
padded rank-local prefix cache, reorders it to global position order, writes it
into the head of dcp_kv_buffer, then appends the new K-nope/K-rope rows.  The
Gloo process group is real; the SGLang parallel context and KV-pool access are
adapted to the test tensors without replacing the gather/reorder implementation.
"""
import json
import os
import statistics
import tempfile
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import kernelgenbench
from kernelgenbench.rows492_501_utils import exact
from kernelgenbench.rows492_501_workers import dcp_extend_worker
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

NAME = "v1_sgl_dcp_extend_kv_gather_"


def _ensure_solution():
    """Register the binding this Verifier run selected (good or negative)."""
    import importlib
    import kernelgenbench as kgb
    if getattr(getattr(kgb, "triton", None), NAME, None) is not None:
        return
    from kernelgenbench import register
    module_name = os.environ.get("KGB_SOLUTION_BINDING",
                                 "kernelgenbench.solution_bindings.v1_sgl_dcp_extend_kv_gather_")
    module = importlib.import_module(module_name)
    register(NAME, NAME, namespace="triton")(getattr(module, NAME))



# world, per-request global prefix lengths, extend rows, K-nope width,
# K-rope width, dtype
COMBOS = [
    (2, (3, 6, 1), 3, 3, 2, torch.float32),
    (3, (1, 4, 8), 2, 5, 3, torch.bfloat16),
    (2, (0, 0), 3, 4, 2, torch.float32),
    (3, (2, 5), 1, 2, 4, torch.float32),
]


def input_build(config):
    """One CPU input bundle per rank; local_indices are request/local-position ordered."""
    world, lengths, extend, dn, dr, dtype = config
    built = []
    for rank in range(world):
        local = [(request, position) for request, length in enumerate(lengths)
                 for position in range(rank, length, world)]
        size = 2 * len(local) + 5
        cache_nope = torch.full((size, 1, dn), -777, dtype=dtype)
        cache_pe = torch.full((size, 1, dr), -888, dtype=dtype)
        indices = torch.arange(len(local), dtype=torch.int64) * 2 + 1
        for row, (request, position) in enumerate(local):
            cache_nope[indices[row], 0] = torch.arange(dn) + 100 * request + 10 * position
            cache_pe[indices[row], 0] = torch.arange(dr) + 100 * request + 10 * position + 50
        extra = 2 if sum(lengths) > 0 else 0
        k_nope = (torch.arange((extend + extra) * dn).reshape(extend + extra, 1, dn) + 300).to(dtype)
        k_pe = (torch.arange((extend + extra) * dr).reshape(extend + extra, 1, dr) + 500).to(dtype)
        built.append(dict(
            cache_nope=cache_nope,
            cache_pe=cache_pe,
            local_indices=indices,
            prefix_lens=lengths,
            k_nope=k_nope,
            k_pe=k_pe,
            dcp_kv_buffer=torch.full((sum(lengths) + extend, 1, dn + dr), -99, dtype=dtype),
        ))
    return built


def expected_output(config):
    _, lengths, extend, dn, dr, dtype = config
    out = torch.empty((sum(lengths) + extend, 1, dn + dr), dtype=dtype)
    row = 0
    for request, length in enumerate(lengths):
        for position in range(length):
            out[row, 0, :dn] = torch.arange(dn) + 100 * request + 10 * position
            out[row, 0, dn:] = torch.arange(dr) + 100 * request + 10 * position + 50
            row += 1
    out[row:, :, :dn] = (torch.arange(extend * dn).reshape(extend, 1, dn) + 300).to(dtype)
    out[row:, :, dn:] = (torch.arange(extend * dr).reshape(extend, 1, dr) + 500).to(dtype)
    return out


def compute_ref(**kw):
    return kernelgenbench.baseline.v1_sgl_dcp_extend_kv_gather_(**kw)


def compute_res(**kw):
    return kernelgenbench.solution.v1_sgl_dcp_extend_kv_gather_(**kw)


def compare(ref_state, res_state, config, rank):
    ref_ret, ref_buffer, ref_inputs = ref_state
    res_ret, res_buffer, res_inputs = res_state
    assert ref_ret is None and res_ret is None
    expected = expected_output(config)
    exact(ref_buffer, expected)
    exact(res_buffer, expected)
    exact(res_buffer, ref_buffer)
    # The cache inputs, local index list and new K rows are read-only.
    for name in ("cache_nope", "cache_pe", "local_indices", "k_nope", "k_pe"):
        exact(res_inputs[name], ref_inputs[name])


def _time_distributed(compute, config, rank, group):
    samples = []
    for _ in range(3):
        arguments = input_build(config)[rank]
        dist.barrier(group)
        start = time.perf_counter()
        compute(**arguments, group=group)
        dist.barrier(group)
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def compute_ref_performance(config, rank, group):
    return _time_distributed(compute_ref, config, rank, group)


def compute_res_performance(config, rank, group):
    return _time_distributed(compute_res, config, rank, group)


@label(NAME)
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_dcp_extend_kv_gather_(config):
    world = config[0]
    with tempfile.TemporaryDirectory(prefix="v1-dcp-extend-") as directory:
        mp.start_processes(
            dcp_extend_worker,
            args=(config, os.path.join(directory, "rdzv"), directory),
            nprocs=world,
            join=True,
            start_method="spawn",
        )
        times = [json.load(open(os.path.join(directory, f"{rank}.json")))
                 for rank in range(world)]
    ref_time = max(pair[0] for pair in times)
    res_time = max(pair[1] for pair in times)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
