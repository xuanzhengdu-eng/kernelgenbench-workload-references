"""Accuracy and performance for v1_sgl_reduce_scatterv.

SGLang ``GroupCoordinator.reduce_scatterv`` contract (official fixed commit):

* rows are the concatenation of all ranks' chunks; ``sizes`` gives the chunk
  length per rank, ``None`` means an equal split;
* the reduction is a sum over ranks, and only this rank's prefix-sum slice is
  returned;
* a supplied ``output`` is shape-checked and written in place, otherwise a new
  tensor is allocated;
* ``input_`` is read-only.

The Solution binds the real pinned ``GroupCoordinator.reduce_scatterv`` method,
running its deterministic branch on the H800 (the pynccl path requires a
multi-rank communicator, which a single visible GPU cannot provide).
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows562_571_utils import fresh, measure, result
from sandbox.verifier.test_parametrize import label, parametrize


# (rows, columns, sizes or None, supplied output, dtype)
COMBOS = [
    (4, 3, [4], False, torch.float32),
    (5, 2, [5], True, torch.bfloat16),
    (6, 4, None, False, torch.float16),
    (3, 1, [3], True, torch.float32),
    (0, 5, [0], False, torch.float32),
    (7, 8, None, True, torch.bfloat16),
]


def input_build(config):
    rows, columns, sizes, supplied, dtype = config
    values = torch.arange(rows * columns, dtype=torch.float32).reshape(rows, columns) / 8
    return {
        "input_": values.to(dtype),
        "output": torch.full((sizes[0] if sizes else rows, columns), -3.0, dtype=dtype)
        if supplied
        else None,
        "sizes": None if sizes is None else list(sizes),
        "group": None,
    }


def compute_ref(**kwargs):
    return kernelgenbench.baseline.v1_sgl_reduce_scatterv(**kwargs)


def compute_res(**kwargs):
    return kernelgenbench.solution.v1_sgl_reduce_scatterv(**kwargs)


def compare(ref_out, res_out, ref_inputs, res_inputs):
    assert isinstance(ref_out, torch.Tensor) and isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == res_inputs["input_"].dtype
    assert res_out.device == res_inputs["input_"].device
    torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0)
    if res_inputs["output"] is not None:
        assert res_out is res_inputs["output"], "supplied output must be written and returned"
        torch.testing.assert_close(
            res_inputs["output"], ref_inputs["output"], rtol=0, atol=0
        )
    torch.testing.assert_close(
        res_inputs["input_"], ref_inputs["input_"], rtol=0, atol=0
    )


def compute_ref_performance(**kwargs):
    return measure(compute_ref, kwargs, kernelgenbench.device)


def compute_res_performance(**kwargs):
    return measure(compute_res, kwargs, kernelgenbench.device)


@label("v1_sgl_reduce_scatterv")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_reduce_scatterv(config):
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, ref_inputs, res_inputs)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return result(ref_time, res_time)
