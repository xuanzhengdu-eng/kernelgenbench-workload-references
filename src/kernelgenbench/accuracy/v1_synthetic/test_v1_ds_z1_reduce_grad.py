"""Accuracy and performance for v1_ds_z1_reduce_grad.

DeepSpeed ZeRO-1 accumulated gradient contract at endBackward, one ``ds_id``:

* ``grad`` and optional ``accumulated`` are same-shaped FP32 tensors.
  ``accumulated`` is updated in place with ``grad``; with ``accumulated=None``
  the saved state aliases ``grad`` itself.
* Only ``update=True`` flushes: every rank averages the accumulated tensor and
  the selected ``[offset:offset+grad_buffer.numel()]`` slice is copied into the
  buffer; the returned state is then ``None``.  With ``update=False`` the
  buffer is untouched and the returned state is the live accumulated tensor.
* ``grad`` is read-only and the returned state aliases the tensor that the
  operator accumulated into.

The Solution is the pinned DeepSpeed DeepCompile operator
``torch.ops.dc.reduce_grad`` registered through ``dc.register_graph_z1``; the
official internal ``grad_tensors_[ds_id].add_`` accumulation and NCCL
all-reduce flush are executed, not a Python re-implementation.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows562_571_utils import fresh, measure, result
from sandbox.verifier.test_parametrize import label, parametrize


# (grad shape, has previous accumulated, update, offset, buffer length,
#  pre-divide flag, non-contiguous grad/accumulated)
COMBOS = [
    ((3, 4), False, False, 0, 6, False, False),
    ((3, 4), True, False, 2, 4, True, False),
    ((3, 4), False, True, 1, 7, False, False),
    ((3, 4), True, True, 5, 7, True, False),
    ((4, 3), True, True, 4, 5, False, True),
    ((4, 3), True, True, 0, 0, True, False),
]


def input_build(config):
    shape, has_acc, update, offset, length, pre_div, strided = config

    def build(phase):
        tensor = (torch.arange(12, dtype=torch.float32).reshape(3, 4) + phase) / 8
        return tensor.t() if strided else tensor.reshape(shape)

    return {
        "grad": build(0),
        "accumulated": build(3) if has_acc else None,
        "grad_buffer": torch.full((length,), -7.0, dtype=torch.float32),
        "offset": offset,
        "update": update,
        "group": None,
        "pre_div_reduce": pre_div,
    }


def compute_ref(**kwargs):
    return kernelgenbench.baseline.v1_ds_z1_reduce_grad(**kwargs)


def compute_res(**kwargs):
    return kernelgenbench.solution.v1_ds_z1_reduce_grad(**kwargs)


def compare(ref_out, res_out, ref_inputs, res_inputs):
    assert isinstance(ref_out, tuple) and len(ref_out) == 2
    assert isinstance(res_out, tuple) and len(res_out) == 2
    for out, side in ((ref_out, ref_inputs), (res_out, res_inputs)):
        empty, state = out
        assert isinstance(empty, torch.Tensor) and empty.shape == (0,)
        assert empty.dtype == side["grad"].dtype and empty.device == side["grad"].device
        if side["update"]:
            assert state is None
        else:
            target = side["grad"] if side["accumulated"] is None else side["accumulated"]
            assert state is target, "returned state must alias the accumulated tensor"
    if res_inputs["update"]:
        assert res_out[1] is None

    torch.testing.assert_close(res_inputs["grad"], ref_inputs["grad"], rtol=0, atol=0)
    if ref_inputs["accumulated"] is not None:
        torch.testing.assert_close(
            res_inputs["accumulated"], ref_inputs["accumulated"], rtol=0, atol=0
        )
        assert res_inputs["accumulated"].stride() == ref_inputs["accumulated"].stride()
    torch.testing.assert_close(
        res_inputs["grad_buffer"], ref_inputs["grad_buffer"], rtol=0, atol=0
    )


def compute_ref_performance(**kwargs):
    return measure(compute_ref, kwargs, kernelgenbench.device)


def compute_res_performance(**kwargs):
    return measure(compute_res, kwargs, kernelgenbench.device)


@label("v1_ds_z1_reduce_grad")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_ds_z1_reduce_grad(config):
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, ref_inputs, res_inputs)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return result(ref_time, res_time)
