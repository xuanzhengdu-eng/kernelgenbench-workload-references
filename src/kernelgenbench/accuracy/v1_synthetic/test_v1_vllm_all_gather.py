"""vLLM device-communicator all-gather: dim 0, last dim and 3-D move-back.

The gathered buffer is dim-0 major; ``movedim(0, dim)`` then flattens the two
adjacent axes.  For dim=1 on a 2-D input the returned tensor is a non-contiguous
view of the gathered buffer, which the comparison checks with strides.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.dist_collective_harness import aggregate, launch, timed
from sandbox.verifier.test_parametrize import label, parametrize

# world size, dim, input rank
COMBOS = [(world, dim, dims)
          for world in (1, 2)
          for dim, dims in ((0, 2), (1, 2), (-1, 2), (1, 3), (-1, 3))]


def input_build(config, rank, dtype=torch.float32):
    world, dim, dims = config
    shape = (2, 3, 4) if dims == 3 else (2, 3)
    total = 1
    for size in shape:
        total *= size
    return {"input_": torch.arange(total, dtype=dtype).reshape(shape) + rank * 10, "dim": dim}


def compute_ref(input_, dim, group):
    return kernelgenbench.baseline.v1_vllm_all_gather(input_, group, dim)


def compute_res(input_, dim, group):
    return kernelgenbench.solution.v1_vllm_all_gather(input_, group, dim)


def compare(config, rank, ref_out, res_out, original, dtype):
    world, dim, dims = config
    axis = dim % original.ndim
    shape = list(original.shape)
    shape[axis] *= world
    expected = torch.cat([input_build(config, peer, dtype)["input_"] for peer in range(world)],
                         dim=axis)
    for out in (ref_out, res_out):
        assert isinstance(out, torch.Tensor)
        assert out.shape == tuple(shape) and out.dtype == dtype
        assert torch.equal(out, expected)
    if world == 1:
        assert torch.equal(ref_out, original) and torch.equal(res_out, original)
    else:
        # A dim-0 gather is a plain view of the gathered buffer; another axis goes
        # through movedim, whose resulting layout is whatever reshape produces.
        if axis == 0:
            assert ref_out.is_contiguous() and res_out.is_contiguous()
        assert res_out.data_ptr() != original.data_ptr()
    # Both sides must agree on the produced layout, contiguous or not.
    assert res_out.is_contiguous() == ref_out.is_contiguous()
    assert res_out.stride() == ref_out.stride()


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


@label("v1_vllm_all_gather")
@parametrize("config", COMBOS)
@parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("config,dtype", [(config, dtype) for config in COMBOS
                                          for dtype in (torch.float32, torch.int64)])
def test_accuracy_v1_vllm_all_gather(config, dtype):
    return aggregate(launch(_worker, config[0], (config, dtype)))
