"""Megatron TP all-reduce: dense input reduced in place, strided input untouched.

``_reduce`` calls ``all_reduce(input_.contiguous())`` and returns the original
handle.  So a dense multi-rank input becomes the group sum, while a strided
multi-rank input keeps its own values (the reduced temporary is thrown away).  A
one-rank group bypasses the collective entirely.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.dist_collective_harness import aggregate, launch, timed
from sandbox.verifier.test_parametrize import label, parametrize

# world size, rows, columns, strided input
COMBOS = [(world, rows, cols, strided)
          for world in (1, 2)
          for rows, cols in ((2, 3), (3, 5))
          for strided in (False, True)]


def input_build(config, rank, dtype=torch.float32):
    world, rows, cols, strided = config
    x = torch.arange(rows * cols, dtype=dtype).reshape(rows, cols) + rank * 10
    if strided:
        wide = torch.full((rows, cols * 2), -1, dtype=dtype)
        wide[:, ::2] = x
        x = wide[:, ::2]
    return {"x": x}


def compute_ref(x, group):
    return kernelgenbench.baseline.v1_megatron_tp_allreduce(x, group)


def compute_res(x, group):
    return kernelgenbench.solution.v1_megatron_tp_allreduce(x, group)


def compare(config, rank, ref_out, res_out, ref_input, res_input, dtype):
    world, rows, cols, strided = config
    if world == 1 or strided:
        expected = input_build(config, rank, dtype)["x"]
    else:
        expected = sum(input_build(config, peer, dtype)["x"] for peer in range(world))
    for out, source in ((ref_out, ref_input), (res_out, res_input)):
        assert isinstance(out, torch.Tensor)
        assert torch.equal(out, expected), "reduced values differ from the documented contract"
        assert out.dtype == dtype and out.shape == (rows, cols)
        # The wrapper returns the input handle, so the storage is shared.
        assert out.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
        assert out.storage_offset() == source.storage_offset()
        assert out.stride() == source.stride()
    assert torch.equal(ref_input, expected) and torch.equal(res_input, expected)


def _worker(rank, config, dtype):
    ref_payload = input_build(config, rank, dtype)
    res_payload = input_build(config, rank, dtype)
    group = torch.distributed.group.WORLD
    ref_out = compute_ref(**ref_payload, group=group)
    res_out = compute_res(**res_payload, group=group)
    compare(config, rank, ref_out, res_out, ref_payload["x"], res_payload["x"], dtype)
    return {"ref_ms": timed(lambda: compute_ref(**ref_payload, group=group)),
            "res_ms": timed(lambda: compute_res(**res_payload, group=group))}


@label("v1_megatron_tp_allreduce")
@parametrize("config", COMBOS)
@parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("config,dtype", [(config, dtype) for config in COMBOS
                                          for dtype in (torch.float32, torch.int64)])
def test_accuracy_v1_megatron_tp_allreduce(config, dtype):
    return aggregate(launch(_worker, config[0], (config, dtype)))
