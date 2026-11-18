"""SUM reduce of every rank's contribution into the root's output buffer.

Reference: ``kernelgenbench.baseline.v1_reduce_to_root_``.
Solution: pinned vLLM ``PyNcclCommunicator.reduce``
(vllm/distributed/device_communicators/pynccl.py#L429-L455), the operator used
by the pynccl-backed reduce path.  A real NCCL communicator is created from a
Gloo CPU group on the H800.

Contract under test:
  * the call returns ``None`` and never mutates ``input_tensor``;
  * only the root's ``output_tensor`` receives the SUM, in place;
  * ``disabled=True`` performs no communication and leaves every output buffer
    untouched, on every rank.
"""
import os

os.environ.setdefault("NCCL_MULTI_RANK_GPU_ENABLE", "1")

from types import SimpleNamespace

import datetime
import json
import tempfile
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

# (rank count, root, shape, disabled)
COMBOS = [
    (2, 0, (4, 7), False),
    (2, 1, (3, 16), False),
    (2, 1, (2, 8), True),
]

_COMM = None


def input_build(config, rank=0):
    world, root, shape, disabled = config
    generator = torch.Generator().manual_seed(20261110 + rank)
    return {
        "output_tensor": torch.full(shape, -99.0),
        "input_tensor": torch.randn(shape, generator=generator),
        "root": root,
        "disabled": disabled,
    }


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def compute_ref(**kwargs):
    return kernelgenbench.baseline.v1_reduce_to_root_(**kwargs)


def compute_res(**kwargs):
    return kernelgenbench.triton.v1_reduce_to_root_(**kwargs)


def compare(ref_out, res_out, ref_inputs, res_inputs, config, rank):
    world, root, shape, disabled = config
    assert ref_out is None and res_out is None
    assert torch.equal(ref_inputs["input_tensor"], res_inputs["input_tensor"])
    if disabled or rank != root:
        # Non-root and disabled ranks keep their pre-filled buffer.
        assert torch.equal(res_inputs["output_tensor"], ref_inputs["output_tensor"])
    else:
        torch.testing.assert_close(
            res_inputs["output_tensor"], ref_inputs["output_tensor"], rtol=0, atol=0
        )


def worker(rank, config, rendezvous, outdir, candidate_bytes):
    # Spawned ranks need the same registered Solution as the parent verifier.
    import cloudpickle
    kernelgenbench.triton = SimpleNamespace(v1_reduce_to_root_=cloudpickle.loads(candidate_bytes))
    world, root, shape, disabled = config
    torch.cuda.set_device(0)
    dist.init_process_group(
        "nccl",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=world,
        timeout=datetime.timedelta(seconds=120),
    )
    try:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        cpu_group = dist.new_group(list(range(world)), backend="gloo")
        global _COMM
        _COMM = PyNcclCommunicator(group=cpu_group, device=torch.device("cuda", 0))
        assert not _COMM.disabled

        ref_inputs = _to_device(input_build(config, rank), "cuda")
        res_inputs = _to_device(input_build(config, rank), "cuda")
        ref_out = compute_ref(**dict(ref_inputs, group=dist.group.WORLD))
        res_out = compute_res(**dict(res_inputs, group=dist.group.WORLD))
        compare(ref_out, res_out, ref_inputs, res_inputs, config, rank)

        if not disabled:
            expected = sum(
                _to_device(input_build(config, r), "cuda")["input_tensor"]
                for r in range(world)
            )
            if rank == root:
                assert torch.equal(res_inputs["output_tensor"], expected)
                assert torch.equal(ref_inputs["output_tensor"], expected)
            else:
                assert torch.equal(res_inputs["output_tensor"],
                                   torch.full(shape, -99.0, device="cuda"))

        def bench(fn):
            inputs = _to_device(input_build(config, rank), "cuda")
            for _ in range(2):
                fn(**dict(inputs, group=dist.group.WORLD))
            torch.cuda.synchronize()
            dist.barrier()
            start = time.perf_counter()
            for _ in range(10):
                fn(**dict(inputs, group=dist.group.WORLD))
            torch.cuda.synchronize()
            dist.barrier()
            return (time.perf_counter() - start) * 1000.0 / 10.0

        ref_time = bench(compute_ref)
        res_time = bench(compute_res)
        with open(os.path.join(outdir, f"{rank}.json"), "w", encoding="utf8") as f:
            json.dump([ref_time, res_time], f)
    finally:
        dist.destroy_process_group()


@label("v1_reduce_to_root_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_reduce_to_root_(config):
    import cloudpickle
    world = config[0]
    with tempfile.TemporaryDirectory(prefix="v1-reduce-root-") as tmp:
        mp.start_processes(
            worker,
            args=(config, os.path.join(tmp, "rdzv"), tmp,
                  cloudpickle.dumps(kernelgenbench.triton.v1_reduce_to_root_)),
            nprocs=world,
            join=True,
            start_method="spawn",
        )
        times = [json.load(open(os.path.join(tmp, f"{rank}.json"), encoding="utf8"))
                 for rank in range(world)]
    ref_time = max(item[0] for item in times)
    res_time = max(item[1] for item in times)
    return CustomBenchmarkResult(
        ref_time=ref_time,
        res_time=res_time,
        speedup=ref_time / res_time if res_time else float("inf"),
    )
