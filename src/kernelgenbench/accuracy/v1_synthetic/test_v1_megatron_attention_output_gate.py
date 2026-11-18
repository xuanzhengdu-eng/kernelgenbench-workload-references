"""Megatron attention output gate; row 26, OP-008e7099223b (Megatron reference).

Separate value x and gate tensors, both the same number of elements. gate is made
contiguous, reshaped to x.shape, sigmoid is evaluated in float32 and the product
is cast back to x.dtype. Cases cover flat/same-shaped/transposed gates, 2-D and
3-D values, three dtypes and an empty tensor. The solution binds
Attention._apply_output_gate, keeping its @jit_fuser wrapper.
"""
import pytest
import torch
import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.activation_rows22_31_checks import (
    config_dtype, fresh, measure, timings,
)
from sandbox.verifier.test_parametrize import label, parametrize


# (shape, dtype, gate_mode): flatten / same-shape / transposed gate storage.
COMBOS = [
    (shape, dtype, gate_mode)
    for shape in [(2, 3, 4), (4, 8), (1, 1, 16), (0, 5)]
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
    for gate_mode in ("flat", "same", "transposed")
]


def input_build(config):
    """x and gate hold the same number of elements; gate storage varies."""
    shape, dtype, gate_mode = config
    rng = torch.Generator(device="cpu").manual_seed(2603)
    x = torch.randn(shape, dtype=dtype, generator=rng)
    gate = torch.randn(shape, dtype=dtype, generator=rng)
    if gate_mode == "flat":
        gate = gate.reshape(-1).contiguous()
    elif gate_mode == "transposed":
        gate = gate.transpose(0, 1).contiguous().transpose(0, 1)
    return {"x": x, "gate": gate}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_megatron_attention_output_gate(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_megatron_attention_output_gate(**inputs)


def compare(ref_out, res_out, dtype):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == dtype
    rtol, atol = {
        torch.float32: (5e-6, 3e-6),
        torch.float16: (2e-3, 1e-3),
        torch.bfloat16: (1e-2, 8e-3),
    }[dtype]
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_megatron_attention_output_gate")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_attention_output_gate(config):
    shape, dtype, gate_mode = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    # The gate is reshaped, not mutated; the value tensor keeps its shape.
    for output, side in ((ref_out, ref_inputs), (res_out, res_inputs)):
        assert output.shape == shape
        assert not torch._C._is_alias_of(output, side["x"])
        assert not torch._C._is_alias_of(output, side["gate"])
    for side in (ref_inputs, res_inputs):
        torch.testing.assert_close(side["x"].cpu(), inputs["x"], rtol=0, atol=0)
        torch.testing.assert_close(side["gate"].cpu(), inputs["gate"], rtol=0, atol=0)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return timings(ref_time, res_time)
