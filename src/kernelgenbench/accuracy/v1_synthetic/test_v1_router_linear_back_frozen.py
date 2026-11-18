"""Accuracy and performance for v1_router_linear_back_frozen.

Operator: Megatron ``linear_with_frozen_weight`` backward (no weight or bias
gradient).

Official source (pinned commit 266f1c97dca477ca8b92c16087506da2000b0b84):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/layers.py#L327-L359

Item: v1_router_linear_back_frozen (723-row 110, frozen branch).  ``compute_res``
runs the official frozen-weight forward under autograd, so the real backward
returns ``(grad_input, None, None, None, None)``.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows102_111_checks as checks


# (input shape, dtype, bias_present)
# O = max(16, H // 2); weight [O, H] is frozen; bias may exist in the forward
# but never receives a gradient.
COMBOS = [
    ((2, 16, 128), torch.float32, True),
    ((2, 16, 128), torch.float16, False),
    ((2, 128, 1024), torch.bfloat16, True),
    ((2, 128, 1024), torch.float32, False),
    ((16, 128), torch.float16, True),
]

TOLERANCE = {
    torch.float32: (3e-5, 3e-5),
    torch.float16: (3e-2, 3e-2),
    torch.bfloat16: (4e-2, 4e-2),
}


def input_build(config):
    shape, dtype, bias_present = config
    out_features = max(16, shape[-1] // 2)
    generator = torch.Generator(device="cpu").manual_seed(20261115)
    grad_output = torch.randn((*shape[:-1], out_features), dtype=dtype,
                              generator=generator)
    weight = torch.randn((out_features, shape[-1]), dtype=dtype, generator=generator)
    bias = (torch.randn((out_features,), dtype=dtype, generator=generator)
            if bias_present else None)
    return {"grad_output": grad_output, "weight": weight, "bias": bias,
            "inp": torch.randn(shape, dtype=dtype, generator=generator)}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_router_linear_back_frozen(
        grad_output=inputs["grad_output"], weight=inputs["weight"])


def compute_res(**inputs):
    return kernelgenbench.solution.v1_router_linear_back_frozen(**inputs)


def compare(ref_out, res_out, dtype):
    assert isinstance(ref_out, tuple) and isinstance(res_out, tuple)
    assert len(ref_out) == len(res_out) == 5
    assert ref_out[0].shape == res_out[0].shape
    assert ref_out[0].dtype == res_out[0].dtype == dtype
    rtol, atol = TOLERANCE[dtype]
    torch.testing.assert_close(res_out[0], ref_out[0], rtol=rtol, atol=atol)
    for index in range(1, 5):
        assert ref_out[index] is None and res_out[index] is None


def compute_ref_performance(**inputs):
    return checks.measure(compute_ref, inputs, device=kernelgenbench.device)


def compute_res_performance(**inputs):
    return checks.measure(compute_res, inputs, device=kernelgenbench.device)


@label("v1_router_linear_back_frozen")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_router_linear_back_frozen(config):
    shape, dtype, bias_present = config
    out_features = max(16, shape[-1] // 2)
    inputs = input_build(config)
    saved = {name: value.clone() for name, value in inputs.items()
             if isinstance(value, torch.Tensor)}
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    for side in (ref_inputs, res_inputs):
        assert side["grad_output"].shape == (*shape[:-1], out_features)
        assert side["weight"].shape == (out_features, shape[-1])
        assert side["inp"].shape == shape

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)
    assert ref_out[0].shape == shape

    for side in (ref_inputs, res_inputs):
        for name in ("grad_output", "weight", "inp"):
            torch.testing.assert_close(side[name].cpu(), saved[name], rtol=0, atol=0)
        if bias_present:
            torch.testing.assert_close(side["bias"].cpu(), saved["bias"], rtol=0, atol=0)
        assert not torch._C._is_alias_of((ref_out if side is ref_inputs else res_out)[0],
                                         side["weight"])

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return checks.timings(ref_time, res_time)
