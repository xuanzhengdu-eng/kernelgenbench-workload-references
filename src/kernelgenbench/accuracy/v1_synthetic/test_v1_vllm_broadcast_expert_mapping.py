"""Elastic-EP expert mapping broadcast: host shape then device values.

Reference: ``kernelgenbench.baseline.v1_vllm_broadcast_expert_mapping``.
Solution: pinned vLLM ``broadcast_expert_mapping``
(vllm/distributed/elastic_ep/elastic_execute.py#L125-L151) driving a real
``StatelessGroupCoordinator``: the shape travels over the TCPStore group and the
int64 mapping over the NCCL device group, exactly as upstream.

Contract under test:
  * the source rank keeps its own tensor (identity) and every other rank gets a
    freshly allocated int64 tensor of the broadcast shape on the given device;
  * values match the source mapping for every rank, including a zero-row map;
  * a non-source rank may pass ``None``;
  * the source tensor is not modified.
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

# (world, source rank, mapping rows, mapping cols)
COMBOS = [
    (1, 0, 2, 4),
    (2, 0, 2, 3),
    (2, 1, 3, 5),
    (2, 1, 0, 3),
]

_COORD = None
_NEGATIVE = False


def input_build(config, rank=0):
    world, src, rows, cols = config
    mapping = None
    if rank == src:
        mapping = torch.arange(rows * cols, dtype=torch.int64).view(rows, cols) + 17
    return {"physical_to_logical": mapping, "src_rank": src, "device": "cpu"}


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def compute_ref(**kwargs):
    return kernelgenbench.baseline.v1_vllm_broadcast_expert_mapping(**kwargs)


def compute_res(**kwargs):
    return kernelgenbench.triton.v1_vllm_broadcast_expert_mapping(**kwargs)


def compare(ref_out, res_out, config, rank, res_input):
    world, src, rows, cols = config
    expected = torch.arange(rows * cols, dtype=torch.int64).view(rows, cols) + 17
    assert torch.is_tensor(ref_out) and torch.is_tensor(res_out)
    assert ref_out.dtype == res_out.dtype == torch.int64
    assert tuple(res_out.shape) == (rows, cols) == tuple(ref_out.shape)
    assert res_out.device.type == ("cuda" if world > 1 else "cpu")
    assert torch.equal(ref_out, expected)
    assert torch.equal(res_out.cpu(), expected)
    if rank == src:
        assert res_out is res_input, "source rank must keep its own tensor"
    else:
        assert res_out is not res_input


def worker(rank, config, rendezvous, outdir, candidate_bytes):
    # Spawned ranks need the same registered Solution as the parent verifier.
    import cloudpickle
    kernelgenbench.triton = SimpleNamespace(v1_vllm_broadcast_expert_mapping=cloudpickle.loads(candidate_bytes))
    world, src, rows, cols = config
    torch.cuda.set_device(0)
    # The reference broadcasts the host shape tensor, so the default process
    # group is Gloo; the solution creates its own NCCL device group.
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=world,
        timeout=datetime.timedelta(seconds=120),
    )
    try:
        from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator

        global _COORD
        _COORD = StatelessGroupCoordinator(
            group_ranks=[list(range(world))],
            local_rank=0,
            torch_distributed_backend="nccl" if world > 1 else "gloo",
            use_device_communicator=world > 1,
            coord_store=dist.distributed_c10d._get_default_store(),
            group_name=f"audit559_w{world}",
            host="127.0.0.1",
            global_rank=rank,
            global_world_size=world,
        )

        ref_inputs = input_build(config, rank)
        res_inputs = input_build(config, rank)
        if rank == src:
            assert ref_inputs["physical_to_logical"] is not None
            assert res_inputs["physical_to_logical"] is not None
            if world > 1:
                # The device group needs a CUDA source tensor.
                res_inputs["physical_to_logical"] = res_inputs[
                    "physical_to_logical"].cuda()
        ref_kwargs = dict(ref_inputs, group=dist.group.WORLD)
        ref_out = compute_ref(**ref_kwargs)
        res_out = compute_res(**res_inputs)
        compare(ref_out, res_out, config, rank, res_inputs["physical_to_logical"])

        def make_kwargs(is_ref):
            inputs = input_build(config, rank)
            if is_ref:
                return dict(inputs, group=dist.group.WORLD)
            if world > 1 and rank == src:
                inputs["physical_to_logical"] = inputs["physical_to_logical"].cuda()
            return inputs

        def bench(fn, is_ref):
            kwargs = make_kwargs(is_ref)
            for _ in range(2):
                fn(**kwargs)
            torch.cuda.synchronize()
            dist.barrier()
            start = time.perf_counter()
            for _ in range(10):
                fn(**kwargs)
            torch.cuda.synchronize()
            dist.barrier()
            return (time.perf_counter() - start) * 1000.0 / 10.0

        ref_time = bench(compute_ref, True)
        res_time = bench(compute_res, False)
        with open(os.path.join(outdir, f"{rank}.json"), "w", encoding="utf8") as f:
            json.dump([ref_time, res_time], f)
    finally:
        dist.destroy_process_group()


@label("v1_vllm_broadcast_expert_mapping")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_broadcast_expert_mapping(config):
    import cloudpickle
    world = config[0]
    with tempfile.TemporaryDirectory(prefix="v1-expert-map-") as tmp:
        mp.start_processes(
            worker,
            args=(config, os.path.join(tmp, "rdzv"), tmp,
                  cloudpickle.dumps(kernelgenbench.triton.v1_vllm_broadcast_expert_mapping)),
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
