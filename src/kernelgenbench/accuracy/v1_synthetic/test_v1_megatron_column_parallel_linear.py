"""Accuracy and performance for v1_megatron_column_parallel_linear.

Item: row 320 (OP-b8a38baad055).  Megatron ``ColumnParallelLinear.forward``:
local block GEMM, optional forward-time weight override with an exact shape
check, gather_output overridable at runtime, and the ``(output, bias)`` return
where the bias is only passed through when ``skip_bias_add`` is set.  The
official pinned module is driven through Gloo ranks on one H800; the report
records that the gather allocation in the pinned mapping targets CUDA, so the
test pins ``torch.cuda.current_device`` to CPU for those ranks.
"""
from kernelgenbench.accuracy.v1_synthetic.pinned_collective_cuda import encode_candidate, bind_candidate
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

# (world, gather_output, runtime override, skip_bias_add, external weight, dtype,
#  has bias)
CASES = [
    (1, False, None, False, False, torch.float32, True),
    (2, False, None, False, False, torch.float32, True),
    (2, True, None, False, False, torch.float32, True),
    (2, False, True, True, True, torch.bfloat16, True),
    (2, True, False, True, False, torch.float32, True),
    (2, True, None, False, True, torch.float32, False),
    (3, True, None, False, False, torch.float32, True),
]
COMBOS = [(case, rank) for case in CASES for rank in range(case[0])]


def input_build(config):
    case, rank = config
    world, gather, override, skip, external, dtype, has_bias = case
    x = torch.randn((4, 2, 8), generator=torch.Generator().manual_seed(272)).to(device="cuda", dtype=dtype)
    weight = torch.randn((6, 8),
                         generator=torch.Generator().manual_seed(70 + rank)).to(device="cuda", dtype=dtype)
    bias = (torch.randn((6,), generator=torch.Generator().manual_seed(90 + rank))
            .to(device="cuda", dtype=dtype)) if has_bias else None
    return {
        "x": x,
        "stored_weight": None if external else weight,
        "weight": weight if external else None,
        "bias": bias,
        "output_size_per_partition": 6,
        "input_size": 8,
        "gather_output": gather,
        "runtime_gather_output": override,
        "skip_bias_add": skip,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_megatron_column_parallel_linear(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_megatron_column_parallel_linear(**inputs)


def compare(ref_out, res_out, dtype):
    assert isinstance(res_out, tuple) and len(res_out) == 2, type(res_out)
    assert isinstance(res_out[0], torch.Tensor)
    assert res_out[0].shape == ref_out[0].shape
    assert res_out[0].dtype == ref_out[0].dtype == dtype
    torch.testing.assert_close(res_out[0], ref_out[0], rtol=0, atol=0)
    if ref_out[1] is None:
        assert res_out[1] is None
    else:
        assert isinstance(res_out[1], torch.Tensor)
        assert res_out[1].shape == ref_out[1].shape
        torch.testing.assert_close(res_out[1], ref_out[1], rtol=0, atol=0)


def worker(rank, case, path, outdir, encoded):
    torch.cuda.set_device(rank)
    bind_candidate(encoded, 'v1_megatron_column_parallel_linear')
    torch.set_num_threads(1)
    dist.init_process_group("nccl", init_method="file://" + path, rank=rank,
                            world_size=case[0],
                            timeout=datetime.timedelta(seconds=120))
    try:
        world, gather, override, skip, external, dtype, has_bias = case
        inputs = input_build((case, rank))
        ref_inputs = {name: (value.clone() if isinstance(value, torch.Tensor) else value)
                      for name, value in inputs.items()}
        res_inputs = {name: (value.clone() if isinstance(value, torch.Tensor) else value)
                      for name, value in inputs.items()}
        ref_out = compute_ref(**ref_inputs, group=dist.group.WORLD)
        res_out = compute_res(**res_inputs, group=dist.group.WORLD)
        compare(ref_out, res_out, dtype)

        effective_gather = gather if override is None else override
        expected_width = 6 * (world if effective_gather else 1)
        assert tuple(res_out[0].shape) == (4, 2, expected_width)

        def local_output(source):
            source_inputs = input_build((case, source))
            block = (source_inputs["weight"] if source_inputs["weight"] is not None
                     else source_inputs["stored_weight"])
            values = source_inputs["x"] @ block.t()
            if not skip and source_inputs["bias"] is not None:
                values = values + source_inputs["bias"]
            return values

        if effective_gather and world > 1:
            expected = torch.cat([local_output(source) for source in range(world)],
                                 dim=-1)
        else:
            expected = local_output(rank)
        torch.testing.assert_close(res_out[0], expected, rtol=0, atol=0)
        if skip and has_bias:
            assert res_out[1].shape == (6,)
            torch.testing.assert_close(res_out[1], inputs["bias"], rtol=0, atol=0)
        else:
            assert res_out[1] is None
        # No input is modified.
        for name in ("x", "stored_weight", "weight", "bias"):
            if isinstance(res_inputs[name], torch.Tensor):
                torch.testing.assert_close(res_inputs[name], inputs[name],
                                           rtol=0, atol=0)

        def bench(fn):
            dist.barrier()
            begin = time.perf_counter()
            for _ in range(5):
                fn(**input_build((case, rank)), group=dist.group.WORLD)
            dist.barrier()
            return (time.perf_counter() - begin) * 1000 / 5

        ref_time = bench(compute_ref)
        res_time = bench(compute_res)

        # Illegal weight selection raises on both sides: no stored weight and no
        # forward override, then an override whose shape differs from the block.
        if case == CASES[0]:
            missing = {name: (value.clone() if isinstance(value, torch.Tensor)
                              else value) for name, value in inputs.items()}
            missing["stored_weight"] = None
            missing["weight"] = None
            for side in (compute_ref, compute_res):
                with pytest.raises(RuntimeError):
                    side(**missing, group=dist.group.WORLD)
            wrong = {name: (value.clone() if isinstance(value, torch.Tensor)
                            else value) for name, value in inputs.items()}
            wrong["weight"] = torch.zeros((5, 8), dtype=dtype, device="cuda")
            for side in (compute_ref, compute_res):
                with pytest.raises(RuntimeError):
                    side(**wrong, group=dist.group.WORLD)
        with open(os.path.join(outdir, str(rank) + ".json"), "w") as handle:
            json.dump([ref_time, res_time], handle)
    finally:
        dist.destroy_process_group()


@label("v1_megatron_column_parallel_linear")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_column_parallel_linear(config):
    case, rank = config
    with tempfile.TemporaryDirectory(prefix="v1-column-parallel-") as directory:
        mp.start_processes(
            worker, args=(case, os.path.join(directory, "rdzv"), directory, encode_candidate('v1_megatron_column_parallel_linear')),
            nprocs=case[0], join=True, start_method="spawn")
        times = [json.load(open(os.path.join(directory, str(r) + ".json")))
                 for r in range(case[0])]
    ref_time = max(entry[0] for entry in times)
    res_time = max(entry[1] for entry in times)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
