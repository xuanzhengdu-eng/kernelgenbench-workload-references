"""Accuracy and performance for v1_ds_update_grad_buffer_.

Operator: DeepCompile native binding dc.update_param_grad_buffer
(DSParamRegistry::updateGradBuffer).
Official source (pinned commit):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/compile/z1.cpp#L130-L134
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/includes/deepcompile.h#L413-L417
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/compile/init.cpp#L95-L97

Item: v1_ds_update_grad_buffer_ (723-row 165). ``compute_ref`` loads this file's
reference from ``kernelgenbench.baseline``; ``compute_res`` calls the real native
binding bound as ``kernelgenbench.solution`` after registering the id through the
same extension.

Inputs are a replacement grad buffer (contiguous, a stride-2 view, and an empty
tensor), an offset, and the registered ds_id. Key assertions: both sides return
None; a non-empty buffer is zeroed in place with its shape, dtype and strides
intact; an empty buffer is accepted; a second replacement on the same id also
returns None and zeroes its own buffer. The registry's internal rebinding is not
externally readable, so it is exercised through repeated updates of the same id.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows162_171_checks as checks


# (shape, offset, non-contiguous view)
SHAPES = [
    ((4, 8), 0, False),
    ((3, 7), 13, True),
    ((0,), 4, False),
]
COMBOS = [(shape, offset, strided, dtype)
          for shape, offset, strided in SHAPES
          for dtype in (torch.float32, torch.bfloat16)]

DS_IDS = {0: 7, 1: 8, 2: 9}


def input_build(config):
    shape, offset, strided, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(20261110)
    raw = torch.randn(shape[:-1] + (shape[-1] * 2 if strided else shape[-1],),
                      dtype=dtype, generator=generator)
    buffer = raw[..., ::2] if strided else raw
    index = SHAPES.index((shape, offset, strided))
    return {
        "grad_buffer": buffer,
        "offset": offset,
        "ds_id": DS_IDS[index],
        "dtype": dtype,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_ds_update_grad_buffer_(
        grad_buffer=inputs["grad_buffer"], offset=inputs["offset"],
        ds_id=inputs["ds_id"])


def compute_res(**inputs):
    return kernelgenbench.solution.v1_ds_update_grad_buffer_(
        grad_buffer=inputs["grad_buffer"], offset=inputs["offset"],
        ds_id=inputs["ds_id"])


def _copy(inputs):
    return {
        "grad_buffer": checks.fresh({"t": inputs["grad_buffer"]},
                                    kernelgenbench.device)["t"],
        "offset": inputs["offset"],
        "ds_id": inputs["ds_id"],
        "dtype": inputs["dtype"],
    }


def compare(ref_out, res_out, dtype=None):
    assert ref_out is None and res_out is None


def _check_zero(output, inputs):
    assert output is None
    buffer = inputs["grad_buffer"]
    assert buffer.numel() == 0 or torch.count_nonzero(buffer).item() == 0, \
        "the replacement buffer must be zeroed in place"
    assert buffer.dtype == inputs["dtype"]


def compute_ref_performance(inputs):
    return checks.measure_factory(lambda: _copy(inputs), compute_ref,
                                  kernelgenbench.device, repeat=10)


def compute_res_performance(inputs):
    return checks.measure_factory(lambda: _copy(inputs), compute_res,
                                  kernelgenbench.device, repeat=10)


@label("v1_ds_update_grad_buffer_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_ds_update_grad_buffer_(config):
    inputs = input_build(config)
    ref_inputs = _copy(inputs)
    res_inputs = _copy(inputs)
    ref_inputs["grad_buffer"].fill_(2.0)
    res_inputs["grad_buffer"].fill_(2.0)
    ref_shape = ref_inputs["grad_buffer"].shape
    ref_stride = ref_inputs["grad_buffer"].stride()
    res_stride = res_inputs["grad_buffer"].stride()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out)
    _check_zero(ref_out, ref_inputs)
    _check_zero(res_out, res_inputs)
    assert ref_inputs["grad_buffer"].shape == ref_shape
    assert ref_inputs["grad_buffer"].stride() == ref_stride
    assert res_inputs["grad_buffer"].stride() == res_stride

    # The registry entry may be rebound repeatedly; a second buffer must be
    # accepted for the same id and zeroed on its own.
    second = checks.fresh({"t": inputs["grad_buffer"]}, kernelgenbench.device)["t"]
    second.fill_(3.0)
    assert compute_res(grad_buffer=second, offset=ref_inputs["offset"],
                       ds_id=res_inputs["ds_id"]) is None
    assert second.numel() == 0 or torch.count_nonzero(second).item() == 0

    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return checks.timings(ref_time, res_time)
