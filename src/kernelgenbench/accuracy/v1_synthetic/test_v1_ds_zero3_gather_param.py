"""Accuracy and performance for v1_ds_zero3_gather_param.

DeepSpeed ZeRO-3 parameter gather contract for equal padded shards:

* every rank contributes one shard of ``ceil(true_numel / world_size)``
  elements; the gathered padded buffer is cropped to ``prod(true_shape)`` and
  reshaped to ``true_shape``;
* the returned dtype is the requested ``target_dtype`` (the shard is cast
  before the gather);
* the local shard is read-only and the result is a fresh tensor.

The Solution is the pinned DeepSpeed DeepCompile operators
``torch.ops.dc.allgather_param`` + ``torch.ops.dc.wait_allgather`` registered
through ``dc.register_z3_param`` / ``dc.register_graph_z3``.  Their real NCCL
all-gather runs; the host-side adapter only performs registration and returns
the operator result.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows562_571_utils import fresh, measure, result
from sandbox.verifier.test_parametrize import label, parametrize


# (world size, true shape, shard dtype, requested dtype).  The job runs one
# rank per process, so the world-size entry documents the contract.
COMBOS = [
    (1, (3, 4), torch.float32, torch.float32),
    (1, (5,), torch.float32, torch.bfloat16),
    (1, (2, 4), torch.bfloat16, torch.float32),
    (1, (4,), torch.float16, torch.float16),
]


def input_build(config):
    world, shape, shard_dtype, target_dtype = config
    numel = 1
    for dim in shape:
        numel *= dim
    size = (numel + world - 1) // world
    full = torch.arange(world * size, dtype=torch.float32)
    return {
        "shard": full[:size].to(shard_dtype),
        "true_shape": tuple(shape),
        "target_dtype": target_dtype,
        "group": None,
    }


def compute_ref(**kwargs):
    return kernelgenbench.baseline.v1_ds_zero3_gather_param(**kwargs)


def compute_res(**kwargs):
    return kernelgenbench.solution.v1_ds_zero3_gather_param(**kwargs)


def compare(ref_out, res_out, ref_inputs, res_inputs):
    assert isinstance(res_out, torch.Tensor) and isinstance(ref_out, torch.Tensor)
    assert res_out.shape == ref_out.shape == tuple(res_inputs["true_shape"])
    assert res_out.dtype == ref_out.dtype == res_inputs["target_dtype"]
    assert res_out.device == res_inputs["shard"].device
    assert res_out.stride() == ref_out.stride()
    assert not torch._C._is_alias_of(res_out, res_inputs["shard"])
    torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(
        res_inputs["shard"], ref_inputs["shard"], rtol=0, atol=0
    )


def compute_ref_performance(**kwargs):
    return measure(compute_ref, kwargs, kernelgenbench.device)


def compute_res_performance(**kwargs):
    return measure(compute_res, kwargs, kernelgenbench.device)


@label("v1_ds_zero3_gather_param")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_ds_zero3_gather_param(config):
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, ref_inputs, res_inputs)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return result(ref_time, res_time)
