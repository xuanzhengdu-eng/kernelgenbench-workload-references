"""DeepSpeed _GatherSequenceLoss.backward: dim-0 window of the gathered loss grad.

The loss was gathered along the token axis, so each rank keeps
``grad_output.narrow(0, rank * local, local)`` with ``local = rows / world``.  The
window is a view over the incoming storage even for a single rank, and the return
is a 3-tuple.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.dist_collective_harness import aggregate, launch, timed
from sandbox.verifier.test_parametrize import label, parametrize

# world size, gathered rows, strided gradient
COMBOS = [
    (world, rows, strided)
    for world in (1, 2)
    for rows in (4, 6)
    for strided in (False, True)
]


def input_build(config, rank, dtype=torch.float32):
    world, rows, strided = config
    grad = torch.arange(rows * 3, dtype=dtype).reshape(rows, 3) + rank * 100
    if strided:
        wide = torch.full((rows, 6), -1, dtype=dtype)
        wide[:, ::2] = grad
        grad = wide[:, ::2]
    return {"grad_output": grad}


def compute_ref(grad_output, group):
    return kernelgenbench.baseline.v1_ds_gather_sequence_loss_backward(grad_output, group)


def compute_res(grad_output, group):
    return kernelgenbench.solution.v1_ds_gather_sequence_loss_backward(grad_output, group)


def compare(config, rank, ref_out, res_out, grad, dtype):
    world, rows, strided = config
    assert isinstance(ref_out, tuple) and isinstance(res_out, tuple)
    assert len(ref_out) == len(res_out) == 3
    assert ref_out[1] is None and ref_out[2] is None
    assert res_out[1] is None and res_out[2] is None
    local = rows // world
    expected = grad.narrow(0, rank * local, local)
    for out in (ref_out[0], res_out[0]):
        assert torch.equal(out, expected)
        assert out.shape == expected.shape and out.dtype == dtype
        # The official backward narrows and does not copy.
        assert out.untyped_storage().data_ptr() == grad.untyped_storage().data_ptr()
        assert out.storage_offset() == rank * local * grad.stride(0)


def _worker(rank, config, dtype):
    payload = input_build(config, rank, dtype)
    grad, group = payload["grad_output"], torch.distributed.group.WORLD
    before = grad.clone()
    ref_out = compute_ref(**payload, group=group)
    res_out = compute_res(**payload, group=group)
    compare(config, rank, ref_out, res_out, grad, dtype)
    assert torch.equal(grad, before)
    return {"ref_ms": timed(lambda: compute_ref(**payload, group=group)),
            "res_ms": timed(lambda: compute_res(**payload, group=group))}


@label("v1_ds_gather_sequence_loss_backward")
@parametrize("config", COMBOS)
@parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("config,dtype", [(config, dtype) for config in COMBOS
                                          for dtype in (torch.float32, torch.int64)])
def test_accuracy_v1_ds_gather_sequence_loss_backward(config, dtype):
    return aggregate(launch(_worker, config[0], (config, dtype)))
