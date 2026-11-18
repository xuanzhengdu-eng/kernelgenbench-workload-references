"""Reduce-scatter backward (sequence-parallel): first-dim all-gather, four slots.

Each rank's gradient is [2, 3] (optionally strided).  The backward concatenates the
peers on dim 0 in rank order, returns four slots whose tail is None, and for a
one-rank group returns the gradient object itself.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.dist_collective_harness import aggregate, timed
from kernelgenbench.accuracy.v1_synthetic.pinned_collective_cuda import launch_cuda
from sandbox.verifier.test_parametrize import label, parametrize

# world size, rows, columns, strided gradient
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
    return {"grad_output": x}


def compute_ref(grad_output, group):
    return kernelgenbench.baseline.v1_megatron_reduce_scatter_backward(grad_output, group)


def compute_res(grad_output, group):
    return kernelgenbench.solution.v1_megatron_reduce_scatter_backward(grad_output, group)


def compare(config, rank, ref_out, res_out, original, dtype):
    world, rows, cols, strided = config
    assert isinstance(ref_out, tuple) and isinstance(res_out, tuple)
    assert len(ref_out) == len(res_out) == 4
    assert all(value is None for value in ref_out[1:])
    assert all(value is None for value in res_out[1:])
    if world == 1:
        assert ref_out[0] is original and res_out[0] is original
        return
    expected = torch.cat([input_build(config, peer, dtype)["grad_output"] for peer in range(world)],
                         dim=0).contiguous()
    for out in (ref_out[0], res_out[0]):
        assert torch.equal(out, expected)
        assert out.shape == (rows * world, cols) and out.dtype == dtype
        assert out.is_contiguous()
        assert out.data_ptr() != original.data_ptr()


def _worker(rank, config, dtype):
    payload = input_build(config, rank, dtype)
    group = torch.distributed.group.WORLD
    before = payload["grad_output"].clone()
    ref_out = compute_ref(**payload, group=group)
    res_out = compute_res(**payload, group=group)
    compare(config, rank, ref_out, res_out, payload["grad_output"], dtype)
    assert torch.equal(payload["grad_output"], before)
    return {"ref_ms": timed(lambda: compute_ref(**payload, group=group)),
            "res_ms": timed(lambda: compute_res(**payload, group=group))}


@label("v1_megatron_reduce_scatter_backward")
@parametrize("config", COMBOS)
@parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("config,dtype", [(config, dtype) for config in COMBOS
                                          for dtype in (torch.float32, torch.int64)])
def test_accuracy_v1_megatron_reduce_scatter_backward(config, dtype):
    return aggregate(launch_cuda(_worker, config[0], (config, dtype), 'v1_megatron_reduce_scatter_backward'))
