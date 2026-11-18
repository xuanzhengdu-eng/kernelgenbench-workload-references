"""Accuracy and performance for v1_megatron_tp_allreduce_backward.

Operator: Megatron-LM ``_ReduceFromModelParallelRegion.backward``.

Official source (pinned commit):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/mappings.py#L217-L245

Contract: returns ``(grad_output, None)``.  The first element must be the very
same tensor object passed in (not a view, clone or cast), the second is the
gradient for the non-tensor process-group argument and is ``None``.  It works for
CPU and CUDA tensors, any floating dtype and non-contiguous strides.  The test
grid crosses shape (including an empty tensor), dtype, strided layout and device.

``compute_ref`` loads the transcribed reference from ``kernelgenbench.baseline``;
``compute_res`` calls the operator bound as ``kernelgenbench.solution``, i.e. the
real pinned Megatron backward during verification.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows702_711_checks as checks


# (rows, columns)
CASES = [(0, 3), (1, 5), (3, 7)]
COMBOS = [
    (case, dtype, strided, device)
    for case in CASES
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
    for strided in (False, True)
    for device in ("cpu", "cuda")
]


def input_build(config):
    """A CPU gradient; the strided variant is a stride-2 column view."""
    (rows, columns), dtype, strided, device = config
    generator = torch.Generator(device="cpu").manual_seed(20261110)
    width = columns * (2 if strided else 1)
    x = torch.randn((rows, width), generator=generator).to(dtype=dtype)
    if strided:
        x = x[:, ::2]
    return {"grad_output": x}


def compute_ref(grad_output):
    return kernelgenbench.baseline.v1_megatron_tp_allreduce_backward(grad_output)


def compute_res(grad_output):
    return kernelgenbench.solution.v1_megatron_tp_allreduce_backward(grad_output)


def compare(ref_out, res_out, ref_inputs, res_inputs):
    """The tuple shape, the per-side object identity and the None group gradient."""
    assert isinstance(ref_out, tuple) and len(ref_out) == 2
    assert isinstance(res_out, tuple) and len(res_out) == 2
    assert ref_out[0] is ref_inputs["grad_output"], \
        "reference must return its own gradient object"
    assert res_out[0] is res_inputs["grad_output"], \
        "solution must return the caller's gradient object"
    assert ref_out[1] is None and res_out[1] is None
    assert res_out[0] is not res_out[1]


def compute_ref_performance(grad_output):
    return checks.measure(
        kernelgenbench.baseline.v1_megatron_tp_allreduce_backward,
        {"grad_output": grad_output},
    )


def compute_res_performance(grad_output):
    return checks.measure(
        kernelgenbench.solution.v1_megatron_tp_allreduce_backward,
        {"grad_output": grad_output},
    )


@label("v1_megatron_tp_allreduce_backward")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_tp_allreduce_backward(config):
    (rows, columns), dtype, strided, device = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, torch.device(device))
    res_inputs = checks.fresh(inputs, torch.device(device))
    assert res_inputs["grad_output"].shape == (rows, columns)
    assert res_inputs["grad_output"].dtype == dtype
    if strided:
        assert res_inputs["grad_output"].stride(-1) == 2

    saved = res_inputs["grad_output"].clone()
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, ref_inputs, res_inputs)
    assert torch.equal(res_inputs["grad_output"], saved), "gradient was modified"

    return checks.timings(compute_ref_performance(**inputs),
                          compute_res_performance(**inputs))
