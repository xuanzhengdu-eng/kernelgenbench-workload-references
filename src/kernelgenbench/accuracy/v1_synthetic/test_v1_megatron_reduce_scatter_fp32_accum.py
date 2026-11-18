"""Reduce-scatter with FP32 accumulation: 1-D output and optional async handle.

Each rank contributes ``world * len(output)`` values.  The group sum is accumulated
in FP32 and downcast into the caller's 1-D buffer.  ``async_op=True`` returns an
object with ``wait()`` and leaves the buffer untouched until that wait;
``async_op=False`` returns None after the copy.  The input mixes +-256 with a small
term, which a low-precision accumulate would lose.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.dist_collective_harness import aggregate, timed
from kernelgenbench.accuracy.v1_synthetic.pinned_collective_cuda import launch_cuda
from sandbox.verifier.test_parametrize import label, parametrize

# world size, output length, async
COMBOS = [(world, length, async_op)
          for world in (2, 3)
          for length in (2, 6)
          for async_op in (False, True)]


def input_build(config, rank, dtype=torch.float32):
    world, length, async_op = config
    values = torch.full((world * length,), 256.0, dtype=dtype, device="cuda")
    values[rank] = 1.0 if rank == 0 else -256.0
    return {"input_": values.reshape(world, length),
            "output_tensor": torch.full((length,), -7, dtype=dtype, device="cuda")}


def compute_ref(input_, group, output_tensor, async_op=False):
    return kernelgenbench.baseline.v1_megatron_reduce_scatter_fp32_accum(
        input_, group, output_tensor, async_op)


def compute_res(input_, group, output_tensor, async_op=False):
    return kernelgenbench.solution.v1_megatron_reduce_scatter_fp32_accum(
        input_, group, output_tensor, async_op)


def compare(config, rank, ref_out, res_out, ref_buffer, res_buffer, dtype):
    world, length, async_op = config
    total = sum(input_build(config, peer, dtype)["input_"].float() for peer in range(world))
    expected = total.reshape(-1).chunk(world, dim=0)[rank].to(dtype)
    for buffer in (ref_buffer, res_buffer):
        assert torch.equal(buffer, expected)
        assert buffer.shape == (length,) and buffer.dtype == dtype
    if async_op:
        assert hasattr(ref_out, "wait") and hasattr(res_out, "wait")
    else:
        assert ref_out is None and res_out is None


def _worker(rank, config, dtype):
    payload = input_build(config, rank, dtype)
    group = torch.distributed.group.WORLD
    ref_payload = dict(payload, output_tensor=payload["output_tensor"].clone(),
                       async_op=config[2])
    res_payload = dict(payload, output_tensor=payload["output_tensor"].clone(),
                       async_op=config[2])
    sentinel = payload["output_tensor"].clone()
    ref_out = compute_ref(**ref_payload, group=group)
    res_out = compute_res(**res_payload, group=group)
    if config[2]:
        # The async handle must not write the buffer before wait().
        assert torch.equal(ref_payload["output_tensor"], sentinel)
        assert torch.equal(res_payload["output_tensor"], sentinel)
        ref_out.wait()
        res_out.wait()
    compare(config, rank, ref_out, res_out, ref_payload["output_tensor"],
            res_payload["output_tensor"], dtype)
    sync = dict(payload, output_tensor=payload["output_tensor"].clone(), async_op=False)
    return {"ref_ms": timed(lambda: compute_ref(**sync, group=group)),
            "res_ms": timed(lambda: compute_res(**sync, group=group))}


@label("v1_megatron_reduce_scatter_fp32_accum")
@parametrize("config", COMBOS)
@parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("config,dtype", [(config, dtype) for config in COMBOS
                                          for dtype in (torch.float32, torch.bfloat16)])
def test_accuracy_v1_megatron_reduce_scatter_fp32_accum(config, dtype):
    return aggregate(launch_cuda(_worker, config[0], (config, dtype), 'v1_megatron_reduce_scatter_fp32_accum'))
