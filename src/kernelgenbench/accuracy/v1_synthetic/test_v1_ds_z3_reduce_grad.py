"""Accuracy and performance for v1_ds_z3_reduce_grad.

DeepSpeed ZeRO-3 endBackward reduce-scatter contract, one ``ds_id``:

* ``grad`` holds ``world_size * grad_buffer.numel()`` flattened elements in
  rank order.  Communication uses the gradient dtype, then the local shard is
  cast to the gradient-buffer dtype before copy/accumulate.
* ``has_acc_grad`` selects accumulate over overwrite.  ``update=True`` clears
  the accumulation flag and invalidates the parameter, so the returned
  ``(new_acc, new_valid)`` is ``(False, False)``; otherwise the incoming
  ``param_valid`` is preserved.
* ``grad`` is read-only and ``grad_buffer`` is mutated in place.

The Solution is the pinned DeepSpeed DeepCompile operator
``torch.ops.dc.reduce_grad`` registered through ``dc.register_graph_z3``; the
real NCCL reduce-scatter, dtype conversion and accumulate path is executed.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows562_571_utils import fresh, measure, result
from sandbox.verifier.test_parametrize import label, parametrize


# (elements per rank, has accumulation, update, pre-divide, grad dtype,
#  buffer dtype, incoming parameter-valid flag)
COMBOS = [
    (7, False, False, False, torch.float32, torch.float32, True),
    (9, True, False, True, torch.float32, torch.float16, True),
    (6, False, True, True, torch.float16, torch.float32, True),
    (8, True, True, False, torch.bfloat16, torch.bfloat16, True),
    (4, False, False, True, torch.float32, torch.bfloat16, False),
    (0, False, False, False, torch.float32, torch.float32, True),
    (0, True, True, True, torch.float16, torch.bfloat16, True),
]


def input_build(config):
    length, has_acc, update, pre_div, grad_dtype, buffer_dtype, valid = config
    grad = (torch.arange(length, dtype=torch.float32) * 0.125).to(grad_dtype)
    buffer = (
        torch.full((length,), 0.25, dtype=buffer_dtype)
        if length
        else torch.empty((0,), dtype=buffer_dtype)
    )
    return {
        "grad": grad,
        "grad_buffer": buffer,
        "has_acc_grad": has_acc,
        "param_valid": valid,
        "update": update,
        "group": None,
        "pre_div_reduce": pre_div,
    }


def compute_ref(**kwargs):
    return kernelgenbench.baseline.v1_ds_z3_reduce_grad(**kwargs)


def compute_res(**kwargs):
    return kernelgenbench.solution.v1_ds_z3_reduce_grad(**kwargs)


def compare(ref_out, res_out, ref_inputs, res_inputs):
    assert isinstance(ref_out, tuple) and len(ref_out) == 3
    assert isinstance(res_out, tuple) and len(res_out) == 3
    for out, side in ((ref_out, ref_inputs), (res_out, res_inputs)):
        empty, acc, valid = out
        assert isinstance(empty, torch.Tensor) and empty.shape == (0,)
        assert empty.dtype == side["grad"].dtype and empty.device == side["grad"].device
        assert isinstance(acc, bool) and isinstance(valid, bool)
    assert res_out[1:] == ref_out[1:]
    assert res_out[1] == (not res_inputs["update"])
    assert res_out[2] == (False if res_inputs["update"] else res_inputs["param_valid"])
    torch.testing.assert_close(res_inputs["grad"], ref_inputs["grad"], rtol=0, atol=0)
    torch.testing.assert_close(
        res_inputs["grad_buffer"], ref_inputs["grad_buffer"], rtol=0, atol=0
    )


def compute_ref_performance(**kwargs):
    return measure(compute_ref, kwargs, kernelgenbench.device)


def compute_res_performance(**kwargs):
    return measure(compute_res, kwargs, kernelgenbench.device)


@label("v1_ds_z3_reduce_grad")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_ds_z3_reduce_grad(config):
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, ref_inputs, res_inputs)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return result(ref_time, res_time)
