"""Megatron last-dim all-gather: rank-ordered rows, concatenated on the last axis.

One rank returns the input object unchanged; two ranks return a fresh contiguous
tensor whose last axis is ``world`` times wider.  A strided input is made
contiguous before the collective, so the caller's tensor is never modified.
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
          for rows, cols in ((2, 3), (3, 4))
          for strided in (False, True)]


def input_build(config, rank, dtype=torch.float32):
    world, rows, cols, strided = config
    x = torch.arange(rows * cols, dtype=dtype, device="cuda").reshape(rows, cols) + rank * 10
    if strided:
        wide = torch.full((rows, cols * 2), -1, dtype=dtype, device="cuda")
        wide[:, ::2] = x
        x = wide[:, ::2]
    return {"input_": x}


def compute_ref(input_, group):
    return kernelgenbench.baseline.v1_megatron_gather_last_dim(input_, group)


def compute_res(input_, group):
    return kernelgenbench.solution.v1_megatron_gather_last_dim(input_, group)


def compare(config, rank, ref_out, res_out, original, dtype):
    world, rows, cols, strided = config
    if world == 1:
        assert ref_out is original and res_out is original, "one rank returns the object"
        return
    expected = torch.cat([input_build(config, peer, dtype)["input_"] for peer in range(world)], dim=-1)
    for out in (ref_out, res_out):
        assert isinstance(out, torch.Tensor)
        assert out.shape == (rows, cols * world) and out.dtype == dtype
        assert out.is_contiguous()
        assert torch.equal(out, expected)
        assert out.data_ptr() != original.data_ptr()


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


@label("v1_megatron_gather_last_dim")
@parametrize("config", COMBOS)
@parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("config,dtype", [(config, dtype) for config in COMBOS
                                          for dtype in (torch.float32, torch.int64)])
def test_accuracy_v1_megatron_gather_last_dim(config, dtype):
    return aggregate(launch_cuda(_worker, config[0], (config, dtype), 'v1_megatron_gather_last_dim'))
