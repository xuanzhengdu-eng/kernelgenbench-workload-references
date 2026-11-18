"""Megatron first-axis SUM reduce-scatter: shard of the group sum.

Input is [4, 6] per rank (optionally strided).  Two ranks return a fresh
``[2, 6]`` contiguous shard equal to ``sum(peers).chunk(2, 0)[rank]``; one rank
returns the input object unchanged.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.dist_collective_harness import aggregate, timed
from kernelgenbench.accuracy.v1_synthetic.pinned_collective_cuda import launch_cuda
from sandbox.verifier.test_parametrize import label, parametrize

# world size, rows, strided input
COMBOS = [(world, rows, strided) for world in (1, 2) for rows in (4, 8) for strided in (False, True)]


def input_build(config, rank, dtype=torch.float32):
    world, rows, strided = config
    x = torch.arange(rows * 6, dtype=dtype, device="cuda").reshape(rows, 6) + rank * 10
    if strided:
        wide = torch.full((rows, 12), -9, dtype=dtype, device="cuda")
        wide[:, ::2] = x
        x = wide[:, ::2]
    return {"input_": x}


def compute_ref(input_, group):
    return kernelgenbench.baseline.v1_megatron_reduce_scatter(input_, group)


def compute_res(input_, group):
    return kernelgenbench.solution.v1_megatron_reduce_scatter(input_, group)


def compare(config, rank, ref_out, res_out, original, dtype):
    world, rows, strided = config
    if world == 1:
        assert ref_out is original and res_out is original
        return
    total = sum(input_build(config, peer, dtype)["input_"] for peer in range(world))
    expected = total.chunk(world, dim=0)[rank]
    for out in (ref_out, res_out):
        assert torch.equal(out, expected)
        assert out.shape == (rows // world, 6) and out.dtype == dtype
        assert out.is_contiguous() and out.data_ptr() != original.data_ptr()


def _worker(rank, config, dtype):
    payload = input_build(config, rank, dtype)
    group = torch.distributed.group.WORLD
    before = payload["input_"].clone()
    ref_out = compute_ref(**payload, group=group)
    res_out = compute_res(**payload, group=group)
    compare(config, rank, ref_out, res_out, payload["input_"], dtype)
    assert torch.equal(payload["input_"], before)
    return {"ref_ms": timed(lambda: compute_ref(**payload, group=group)),
            "res_ms": timed(lambda: compute_res(**payload, group=group))}


@label("v1_megatron_reduce_scatter")
@parametrize("config", COMBOS)
@parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("config,dtype", [(config, dtype) for config in COMBOS
                                          for dtype in (torch.float32, torch.int64)])
def test_accuracy_v1_megatron_reduce_scatter(config, dtype):
    return aggregate(launch_cuda(_worker, config[0], (config, dtype), 'v1_megatron_reduce_scatter'))
