"""Megatron last-axis SUM reduce-scatter: shard plus a fresh tensor even at one rank.

The flattened split/cat makes the one-rank path allocate, so the result is never
the input object.  Two ranks return ``[4, 3]`` from a ``[4, 6]`` input.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.dist_collective_harness import aggregate, timed
from kernelgenbench.accuracy.v1_synthetic.pinned_collective_cuda import launch_cuda
from sandbox.verifier.test_parametrize import label, parametrize

# world size, rows, columns, strided input
COMBOS = [(world, rows, cols, strided)
          for world in (1, 2)
          for rows, cols in ((4, 6), (2, 8))
          for strided in (False, True)]


def input_build(config, rank, dtype=torch.float32):
    world, rows, cols, strided = config
    x = torch.arange(rows * cols, dtype=dtype, device="cuda").reshape(rows, cols) + rank * 10
    if strided:
        wide = torch.full((rows, cols * 2), -9, dtype=dtype, device="cuda")
        wide[:, ::2] = x
        x = wide[:, ::2]
    return {"input_": x}


def compute_ref(input_, group):
    return kernelgenbench.baseline.v1_megatron_reduce_scatter_last(input_, group)


def compute_res(input_, group):
    return kernelgenbench.solution.v1_megatron_reduce_scatter_last(input_, group)


def compare(config, rank, ref_out, res_out, original, dtype):
    world, rows, cols, strided = config
    total = sum(input_build(config, peer, dtype)["input_"] for peer in range(world))
    expected = total.chunk(world, dim=-1)[rank].contiguous()
    for out in (ref_out, res_out):
        assert torch.equal(out, expected)
        assert out.shape == (rows, cols // world) and out.dtype == dtype
        assert out.is_contiguous()
        assert out.data_ptr() != original.data_ptr(), "last-axis path always allocates"
    assert ref_out.data_ptr() != res_out.data_ptr()


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


@label("v1_megatron_reduce_scatter_last")
@parametrize("config", COMBOS)
@parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("config,dtype", [(config, dtype) for config in COMBOS
                                          for dtype in (torch.float32, torch.int64)])
def test_accuracy_v1_megatron_reduce_scatter_last(config, dtype):
    return aggregate(launch_cuda(_worker, config[0], (config, dtype), 'v1_megatron_reduce_scatter_last'))
