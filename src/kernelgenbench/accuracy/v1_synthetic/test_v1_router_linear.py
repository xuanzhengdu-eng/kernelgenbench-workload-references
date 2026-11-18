"""Accuracy and performance for v1_router_linear (Megatron).

Operator: Megatron ``RouterGatingLinearFunction`` forward, router dtype.

Official source (pinned commit 266f1c97dca477ca8b92c16087506da2000b0b84):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/moe_utils.py#L1299-L1336

Item: v1_router_linear (723-row 111), Megatron reference.  The SGLang linear and
patch branches are separate official entries with their own references/tests.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows102_111_checks as checks


# (input shape, dtype, bias_present, router_dtype)
# O = max(16, H // 2); inp is [..., H], weight [O, H], bias [O] or None.
COMBOS = [
    ((2, 16, 128), torch.float32, True, torch.float32),
    ((2, 16, 128), torch.float16, True, torch.float32),
    ((2, 16, 128), torch.bfloat16, False, torch.float32),
    ((2, 128, 1024), torch.float32, True, torch.float32),
    ((2, 128, 1024), torch.bfloat16, True, torch.float32),
    ((2, 128, 1024), torch.float32, False, torch.float64),
    ((16, 128), torch.float16, True, torch.float32),
]

TOLERANCE = {
    torch.float32: (4e-3, 5e-2),
    torch.float64: (2e-5, 2e-5),
    torch.float16: (3e-2, 3e-2),
    torch.bfloat16: (4e-2, 4e-2),
}


def input_build(config):
    shape, dtype, bias_present, router_dtype = config
    out_features = max(16, shape[-1] // 2)
    generator = torch.Generator(device="cpu").manual_seed(20261115)
    inp = torch.randn(shape, dtype=dtype, generator=generator)
    weight = torch.randn((out_features, shape[-1]), dtype=dtype, generator=generator)
    # The bias is supplied in router_dtype: the official fallback upcasts it
    # with ``bias.to(router_dtype)`` before the add, and the TE 2.9.0 build in
    # this environment mis-applies a low-precision bias when out_dtype is fp32
    # (measured max abs error 3.9 for bf16 bias vs 7.6e-6 for fp32 bias), so the
    # bias dtype is fixed to the documented add dtype here.
    bias = (torch.randn((out_features,), dtype=router_dtype, generator=generator)
            if bias_present else None)
    return {"inp": inp, "weight": weight, "bias": bias,
            "router_dtype": router_dtype}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_router_linear(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_router_linear(**inputs)


def compare(ref_out, res_out, dtype, router_dtype):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == router_dtype
    rtol, atol = TOLERANCE[dtype]
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)


def compute_ref_performance(**inputs):
    return checks.measure(compute_ref, inputs, device=kernelgenbench.device)


def compute_res_performance(**inputs):
    return checks.measure(compute_res, inputs, device=kernelgenbench.device)


@label("v1_router_linear")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_router_linear(config):
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
        if bias_present:
            assert side["bias"].shape == (out_features,)
            assert side["bias"].dtype == router_dtype
        else:
            assert side["bias"] is None

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype, router_dtype)
    assert res_out.shape == (*shape[:-1], out_features)

    for side in (ref_inputs, res_inputs):
        for name in ("inp", "weight"):
            torch.testing.assert_close(side[name].cpu(), saved[name], rtol=0, atol=0)
        if bias_present:
            torch.testing.assert_close(side["bias"].cpu(), saved["bias"], rtol=0, atol=0)
        assert not torch._C._is_alias_of(res_out, side["inp"])
        assert not torch._C._is_alias_of(res_out, side["weight"])

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return checks.timings(ref_time, res_time)
