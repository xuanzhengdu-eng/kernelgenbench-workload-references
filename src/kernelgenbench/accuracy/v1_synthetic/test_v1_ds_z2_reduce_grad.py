"""Accuracy and performance for v1_ds_z2_reduce_grad.

DeepSpeed ZeRO-2 endBackward reduce/accumulate contract, one ``ds_id``:

* ``grad`` is flattened and averaged over the process group (world size 1 in
  this run, so the average is the identity), then the slice
  ``[offset:offset+grad_buffer.numel()]`` is copied into ``grad_buffer`` when
  ``has_acc_grad`` is false and added to it otherwise.
* ``update=True`` clears the accumulation state after the flush, so the next
  reduction overwrites instead of accumulating.
* ``grad`` is read-only; ``grad_buffer`` is mutated in place.  The returned
  zero-length tensor has the dtype/device of ``grad``.

The Solution is the pinned DeepSpeed DeepCompile operator
``torch.ops.dc.reduce_grad`` (source ``csrc/compile/z2.cpp``) registered through
``dc.register_graph_z2``; its real NCCL all-reduce and copy/accumulate path is
what the comparison below exercises.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows562_571_utils import fresh, measure, result
from sandbox.verifier.test_parametrize import label, parametrize


# (grad shape, has previous accumulation, update, offset, buffer length,
#  pre-divide flag, non-contiguous grad)
COMBOS = [
    ((3, 4), False, False, 0, 6, False, False),
    ((3, 4), True, False, 2, 4, True, False),
    ((3, 4), False, True, 1, 7, False, False),
    ((3, 4), True, True, 5, 7, True, False),
    ((4, 3), True, False, 4, 5, False, True),
    ((3, 4), False, False, 0, 0, True, False),
    ((3, 4), True, True, 0, 0, False, False),
    ((3, 4), True, False, 0, 0, False, False),
]


def input_build(config):
    shape, has_acc, update, offset, length, pre_div, strided = config
    grad = (torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(shape) + 1) / 8
    if strided:
        grad = grad.t()
    return {
        "grad": grad,
        "grad_buffer": torch.full((length,), 0.5, dtype=torch.float32),
        "offset": offset,
        "has_acc_grad": has_acc,
        "update": update,
        "group": None,
        "pre_div_reduce": pre_div,
    }


def compute_ref(**kwargs):
    return kernelgenbench.baseline.v1_ds_z2_reduce_grad(**kwargs)


def compute_res(**kwargs):
    return kernelgenbench.solution.v1_ds_z2_reduce_grad(**kwargs)


def compare(ref_out, res_out, ref_inputs, res_inputs):
    assert isinstance(ref_out, tuple) and len(ref_out) == 2
    assert isinstance(res_out, tuple) and len(res_out) == 2
    for out, side in ((ref_out, ref_inputs), (res_out, res_inputs)):
        empty, flag = out
        assert isinstance(empty, torch.Tensor) and empty.shape == (0,)
        assert empty.dtype == side["grad"].dtype and empty.device == side["grad"].device
        assert isinstance(flag, bool)
    assert res_out[1] == ref_out[1]
    assert res_out[1] == (
        False if res_inputs["update"]
        else (res_inputs["has_acc_grad"] or res_inputs["grad_buffer"].numel() > 0)
    )
    # grad is read-only, stride layout included.
    torch.testing.assert_close(res_inputs["grad"], ref_inputs["grad"], rtol=0, atol=0)
    assert res_inputs["grad"].stride() == ref_inputs["grad"].stride()
    # The official operator writes exactly the same buffer bytes.
    torch.testing.assert_close(
        res_inputs["grad_buffer"], ref_inputs["grad_buffer"], rtol=0, atol=0
    )


def compute_ref_performance(**kwargs):
    return measure(compute_ref, kwargs, kernelgenbench.device)


def compute_res_performance(**kwargs):
    return measure(compute_res, kwargs, kernelgenbench.device)


@label("v1_ds_z2_reduce_grad")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_ds_z2_reduce_grad(config):
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    assert res_inputs["grad"].is_contiguous() == (not config[6])

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, ref_inputs, res_inputs)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return result(ref_time, res_time)
