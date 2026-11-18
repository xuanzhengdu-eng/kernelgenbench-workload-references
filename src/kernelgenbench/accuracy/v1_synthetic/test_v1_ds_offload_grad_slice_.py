"""Accuracy and performance for v1_ds_offload_grad_slice_.

Operator: ZenFlowZeroOptimizerParallel.async_inplace_copy_grad_to_fp32_buffer_from_gpu
(native torch).
Official source (pinned commit):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/runtime/zenflow/zenflow_stage_1_and_2.py#L678-L693

Item: v1_ds_offload_grad_slice_ (723-row 163). ``compute_ref`` loads this file's
reference from ``kernelgenbench.baseline``; ``compute_res`` calls the real official
method bound as ``kernelgenbench.solution``.

Inputs are a parameter holding a 12-element gradient, a (3, 4) gradient view and a
(4, 4) destination prefilled with -3, plus explicit source/destination offsets and
the master dtype. Cases cover same-dtype copy, bf16/fp16 source cast to fp32,
zero elements, and an fp32 source cast down to fp16. Key assertions: both sides
return None, the destination slice equals the cast source and only that slice
changes, the gradient and parameter data are untouched, and ``param.grad`` is
cleared.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows162_171_checks as checks


# (source dtype, destination dtype, source offset, destination offset, count,
# master dtype)
COMBOS = [
    (torch.float32, torch.float32, 2, 3, 5, torch.float32),
    (torch.bfloat16, torch.float32, 0, 4, 6, torch.float32),
    (torch.float16, torch.float32, 4, 1, 0, torch.float32),
    (torch.float32, torch.float16, 1, 2, 7, torch.float16),
]

TOLERANCE = {
    torch.float32: (0.0, 0.0),
    torch.float16: (0.0, 0.0),
    torch.bfloat16: (0.0, 0.0),
}


def input_build(config):
    dtype, out_dtype, source_offset, dest_offset, count, master_dtype = config
    generator = torch.Generator(device="cpu").manual_seed(6742)
    param = torch.nn.Parameter(torch.randn((12,), generator=generator, dtype=dtype))
    param.grad = torch.randn((12,), generator=generator, dtype=dtype)
    gradient = torch.randn((3, 4), generator=generator, dtype=dtype)
    destination = torch.full((4, 4), -3.0, dtype=out_dtype)
    return {
        "param": param,
        "gradient": gradient,
        "destination": destination,
        "source_offset": source_offset,
        "dest_offset": dest_offset,
        "num_elements": count,
        "master_dtype": master_dtype,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_ds_offload_grad_slice_(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_ds_offload_grad_slice_(**inputs)


def _copy(inputs):
    device = kernelgenbench.device
    param = torch.nn.Parameter(inputs["param"].detach().clone().to(device))
    param.grad = inputs["param"].grad.detach().clone().to(device)
    return {
        "param": param,
        "gradient": inputs["gradient"].clone().to(device),
        "destination": inputs["destination"].clone().to(device),
        "source_offset": inputs["source_offset"],
        "dest_offset": inputs["dest_offset"],
        "num_elements": inputs["num_elements"],
        "master_dtype": inputs["master_dtype"],
    }


def compare(ref_out, res_out, dtype=None):
    assert ref_out is None and res_out is None


def _check_side(output, inputs, config):
    dtype, out_dtype, source_offset, dest_offset, count, master_dtype = config
    assert output is None
    assert inputs["param"].grad is None, "offload must clear param.grad"
    expected = inputs["gradient"].view(-1).narrow(0, source_offset, count)
    if expected.dtype != master_dtype:
        expected = expected.to(master_dtype)
    flat = inputs["destination"].view(-1)
    torch.testing.assert_close(flat.narrow(0, dest_offset, count), expected,
                               rtol=0, atol=0)
    untouched = torch.ones_like(flat, dtype=torch.bool)
    untouched.narrow(0, dest_offset, count).fill_(False)
    torch.testing.assert_close(flat[untouched],
                               torch.full_like(flat[untouched], -3.0), rtol=0, atol=0)
    assert inputs["destination"].dtype == out_dtype


def compute_ref_performance(inputs):
    return checks.measure_factory(lambda: _copy(inputs), compute_ref,
                                  kernelgenbench.device)


def compute_res_performance(inputs):
    return checks.measure_factory(lambda: _copy(inputs), compute_res,
                                  kernelgenbench.device)


@label("v1_ds_offload_grad_slice_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_ds_offload_grad_slice_(config):
    inputs = input_build(config)
    ref_inputs = _copy(inputs)
    res_inputs = _copy(inputs)
    ref_gradient = ref_inputs["gradient"].clone()
    res_gradient = res_inputs["gradient"].clone()
    ref_param = ref_inputs["param"].detach().clone()
    res_param = res_inputs["param"].detach().clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out)
    _check_side(ref_out, ref_inputs, config)
    _check_side(res_out, res_inputs, config)

    torch.testing.assert_close(ref_inputs["gradient"], ref_gradient, rtol=0, atol=0)
    torch.testing.assert_close(res_inputs["gradient"], res_gradient, rtol=0, atol=0)
    torch.testing.assert_close(ref_inputs["param"].detach(), ref_param, rtol=0, atol=0)
    torch.testing.assert_close(res_inputs["param"].detach(), res_param, rtol=0, atol=0)
    torch.testing.assert_close(ref_inputs["destination"], res_inputs["destination"],
                               rtol=0, atol=0)

    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return checks.timings(ref_time, res_time)
