"""Sum DP contributions and scatter variable rows into the caller's output.

Reference: ``kernelgenbench.baseline.v1_vllm_combine_into_output_``.
Solution: pinned vLLM ``GroupCoordinator.reduce_scatterv_into_output`` -- the
operator invoked by ``AgRsAll2AllManager.combine_into_output``
(vllm/distributed/device_communicators/all2all.py#L183-L198).  A real 2-rank
NCCL group is used on the H800.

Contract under test:
  * every rank passes a full ``sum(sizes)`` row tensor;
  * the returned object is the caller's ``output`` buffer (in-place);
  * root-local rows equal the sum over ranks of the matching slice;
  * the input tensor is left untouched.
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

# (per-rank sizes, width, dtype)
COMBOS = [
    ((4, 4), 4, torch.float32),
    ((2, 3), 4, torch.float32),
    ((0, 2), 3, torch.float32),
    ((1, 1), 8, torch.int32),
    ((2, 3), 4, torch.int32),
]

_COORD = None


def input_build(config, rank=0):
    sizes, width, dtype = config
    rows = sum(sizes)
    hidden = (torch.arange(rows * width, dtype=torch.float32) + 10 * rank).view(rows, width)
    return {
        "hidden_states": hidden.to(dtype),
        "output": torch.full((sizes[rank], width), -1.0, dtype=torch.float32).to(dtype),
        "sizes": list(sizes),
    }


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def compute_ref(**kwargs):
    return kernelgenbench.baseline.v1_vllm_combine_into_output_(**kwargs)


def compute_res(**kwargs):
    return kernelgenbench.triton.v1_vllm_combine_into_output_(**kwargs)


def compare(ref_out, res_out, ref_inputs, res_inputs, config):
    sizes, width, dtype = config
    # Upstream returns a dim-0 contiguous view that aliases the caller buffer.
    assert torch._C._is_alias_of(res_out, res_inputs["output"])
    assert torch._C._is_alias_of(ref_out, ref_inputs["output"])
    assert res_out.data_ptr() == res_inputs["output"].data_ptr()
    assert res_out.dtype == dtype and res_out.shape == ref_out.shape
    torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0)


def _bench(fn, kwargs):
    for _ in range(2):
        fn(**kwargs)
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    for _ in range(5):
        fn(**kwargs)
    torch.cuda.synchronize()
    dist.barrier()
    return (time.perf_counter() - start) * 1000.0 / 5.0


def worker(rank, config, rendezvous, outdir, candidate_bytes):
    # Spawned ranks need the same registered Solution as the parent verifier.
    import cloudpickle
    kernelgenbench.triton = SimpleNamespace(v1_vllm_combine_into_output_=cloudpickle.loads(candidate_bytes))
    torch.cuda.set_device(0)
    world = len(config[0])
    dist.init_process_group(
        "nccl",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=world,
        timeout=datetime.timedelta(seconds=120),
    )
    try:
        from vllm.distributed.parallel_state import GroupCoordinator

        global _COORD
        _COORD = GroupCoordinator(
            [list(range(world))],
            local_rank=0,
            torch_distributed_backend="nccl",
            use_device_communicator=True,
            group_name=f"audit553_w{world}",
        )

        ref_inputs = _to_device(input_build(config, rank), "cuda")
        res_inputs = _to_device(input_build(config, rank), "cuda")
        ref_hidden_before = ref_inputs["hidden_states"].clone()
        res_hidden_before = res_inputs["hidden_states"].clone()

        ref_kwargs = dict(ref_inputs, group=dist.group.WORLD)
        res_kwargs = dict(res_inputs, group=dist.group.WORLD)
        ref_out = compute_ref(**ref_kwargs)
        res_out = compute_res(**res_kwargs)
        compare(ref_out, res_out, ref_inputs, res_inputs, config)

        # Independent expectation: sum matching slices over ranks.
        expected = sum(
            _to_device(input_build(config, r)["hidden_states"], "cuda")
            for r in range(world)
        )[sum(config[0][:rank]):sum(config[0][:rank]) + config[0][rank]]
        assert torch.equal(res_out, expected), (res_out, expected)
        assert torch.equal(res_inputs["output"], expected), "solution did not write output"
        assert torch.equal(ref_inputs["output"], expected), "reference did not write output"
        assert torch.equal(ref_inputs["hidden_states"], ref_hidden_before)
        assert torch.equal(res_inputs["hidden_states"], res_hidden_before)

        ref_time = _bench(compute_ref, ref_kwargs)
        res_time = _bench(compute_res, res_kwargs)
        with open(os.path.join(outdir, f"{rank}.json"), "w", encoding="utf8") as f:
            json.dump([ref_time, res_time], f)
    finally:
        dist.destroy_process_group()


@label("v1_vllm_combine_into_output_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_combine_into_output_(config):
    import cloudpickle
    world = len(config[0])
    with tempfile.TemporaryDirectory(prefix="v1-vllm-combine-") as tmp:
        mp.start_processes(
            worker,
            args=(config, os.path.join(tmp, "rdzv"), tmp,
                  cloudpickle.dumps(kernelgenbench.triton.v1_vllm_combine_into_output_)),
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
