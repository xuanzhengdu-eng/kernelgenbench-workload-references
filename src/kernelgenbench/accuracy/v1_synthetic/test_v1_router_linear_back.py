"""Accuracy and performance for v1_router_linear_back.

Operator: Megatron ``RouterGatingLinearFunction`` backward.

Official source (pinned commit 266f1c97dca477ca8b92c16087506da2000b0b84):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/moe_utils.py#L1339-L1373

Item: v1_router_linear_back (723-row 110), router backward.  The frozen-weight
branch is a different official entry (``linear_with_frozen_weight``) and has its
own reference/test.  ``compute_res`` runs the official forward under autograd and
takes ``torch.autograd.grad``, so the real backward runs.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows102_111_checks as checks


# (input shape, dtype, bias_present, router_dtype)
# O = max(16, H // 2) for the synthetic router; inp is [..., H], weight [O, H],
# bias [O], grad_output [..., O].
COMBOS = [
    ((2, 16, 128), torch.float32, True, torch.float32),
    ((2, 16, 128), torch.float16, True, torch.float32),
    ((2, 16, 128), torch.bfloat16, False, torch.float32),
    ((2, 128, 1024), torch.float32, True, torch.float32),
    ((2, 128, 1024), torch.float16, False, torch.float32),
    ((2, 128, 1024), torch.bfloat16, True, torch.float64),
    ((16, 128), torch.float32, True, torch.float32),
]

TOLERANCE = {
    torch.float32: (4e-3, 5e-2),
    torch.float16: (3e-2, 3e-2),
    torch.bfloat16: (4e-2, 4e-2),
}


def input_build(config):
    """inp [..., H], weight [O, H], bias [O] or None, grad_output [..., O]."""
    shape, dtype, bias_present, router_dtype = config
    out_features = max(16, shape[-1] // 2)
    generator = torch.Generator(device="cpu").manual_seed(20261115)
    inp = torch.randn(shape, dtype=dtype, generator=generator)
    weight = torch.randn((out_features, shape[-1]), dtype=dtype, generator=generator)
    bias = (torch.randn((out_features,), dtype=dtype, generator=generator)
            if bias_present else None)
    grad_output = torch.randn((*shape[:-1], out_features), dtype=router_dtype,
                              generator=generator)
    return {
        "grad_output": grad_output,
        "inp": inp,
        "weight": weight,
        "bias": bias,
        "router_dtype": router_dtype,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_router_linear_back(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_router_linear_back(**inputs)


def compare(ref_out, res_out, dtype):
    assert isinstance(ref_out, tuple) and isinstance(res_out, tuple)
    assert len(ref_out) == len(res_out) == 4
    rtol, atol = TOLERANCE[dtype]
    for index, (expected, actual) in enumerate(zip(ref_out, res_out)):
        if expected is None or actual is None:
            assert expected is None and actual is None, f"tuple item {index} not both None"
            continue
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype == dtype
        if expected.numel():
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def compute_ref_performance(**inputs):
    return checks.measure(compute_ref, inputs, device=kernelgenbench.device)


def compute_res_performance(**inputs):
    return checks.measure(compute_res, inputs, device=kernelgenbench.device)


@label("v1_router_linear_back")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_router_linear_back(config):
    shape, dtype, bias_present, router_dtype = config
    out_features = max(16, shape[-1] // 2)
    inputs = input_build(config)
    saved = {name: value.clone() for name, value in inputs.items()
             if isinstance(value, torch.Tensor)}
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    for side in (ref_inputs, res_inputs):
        assert side["inp"].shape == shape
        assert side["weight"].shape == (out_features, shape[-1])
        assert side["grad_output"].shape == (*shape[:-1], out_features)
        assert side["grad_output"].dtype == router_dtype
        if bias_present:
            assert side["bias"].shape == (out_features,)
        else:
            assert side["bias"] is None

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    # Optional-bias contract and the fixed four-item tuple.
    assert (ref_out[2] is None) == (not bias_present)
    assert (res_out[2] is None) == (not bias_present)
    assert ref_out[3] is None and res_out[3] is None
    assert ref_out[0].shape == shape
    assert ref_out[1].shape == (out_features, shape[-1])
    if bias_present:
        assert ref_out[2].shape == (out_features,)

    for side in (ref_inputs, res_inputs):
        for name in ("inp", "weight", "grad_output"):
            torch.testing.assert_close(side[name].cpu(), saved[name], rtol=0, atol=0)
        if bias_present:
            torch.testing.assert_close(side["bias"].cpu(), saved["bias"], rtol=0, atol=0)
        for grad in (ref_out if side is ref_inputs else res_out)[:2]:
            if grad is not None:
                assert not torch._C._is_alias_of(grad, side["inp"])
                assert not torch._C._is_alias_of(grad, side["weight"])

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return checks.timings(ref_time, res_time)
