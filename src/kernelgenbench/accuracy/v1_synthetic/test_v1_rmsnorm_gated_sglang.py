"""Accuracy and performance for v1_rmsnorm_gated_sglang (row 65, SGLang branch).

Contract (SGLang FLA RMSNorm, norm_before_gate=False): with a gate, activation is
SiLU and the gate multiplies x *before* the RMS statistic; ``gate=None`` degrades
to a plain RMSNorm.  The fp32 affine weight is applied afterwards and the result
is cast to the input dtype.
Official: https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/attention/fla/layernorm_gated.py#L41-L72
Module: https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/attention/fla/layernorm_gated.py#L488-L530
"""
import pytest
import torch
import torch.nn.functional as F

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows62_71_checks import (
    fresh, measure, result, tolerance, unchanged,
)
from sandbox.verifier.test_parametrize import label, parametrize

# (shape, dtype, has_gate, group_size, weight_fp32)
COMBOS = [
    (shape, dtype, has_gate, group_size, weight_fp32)
    for shape in [(2, 16), (2, 3, 32), (7, 128)]
    for dtype in (torch.float32, torch.bfloat16)
    for has_gate in (True, False)
    for group_size in (None, 8)
    for weight_fp32 in (False, True)
    if group_size is None or shape[-1] % group_size == 0
]


def input_build(config):
    shape, dtype, has_gate, _, weight_fp32 = config
    generator = torch.Generator(device="cpu").manual_seed(6511)
    x = torch.randn(shape, dtype=dtype, generator=generator)
    gate = torch.randn(shape, dtype=dtype, generator=generator) if has_gate else None
    weight = torch.randn((shape[-1],), dtype=torch.float32 if weight_fp32 else dtype,
                         generator=generator)
    return {"x": x, "gate": gate, "weight": weight, "eps": 1e-5}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_rmsnorm_gated_sglang(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_rmsnorm_gated_sglang(**inputs)


def compare(ref_out, res_out, dtype):
    assert isinstance(res_out, torch.Tensor) and isinstance(ref_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == dtype
    rtol, atol = tolerance(dtype)
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)


def plain_norm(value, eps, group_size):
    """Inline fp32 RMS normalization used by the contract checks."""
    if group_size is None:
        return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
    grouped = value.reshape(*value.shape[:-1], value.shape[-1] // group_size, group_size)
    return (grouped * torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + eps)).reshape(
        value.shape)


@label("v1_rmsnorm_gated_sglang")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_rmsnorm_gated_sglang(config):
    shape, dtype, has_gate, group_size, _ = config
    inputs = input_build(config)
    before = {k: v.clone() for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)

    ref_out = kernelgenbench.baseline.v1_rmsnorm_gated_sglang(**ref_inputs, group_size=group_size)
    res_out = kernelgenbench.solution.v1_rmsnorm_gated_sglang(**res_inputs, group_size=group_size)
    compare(ref_out, res_out, dtype)
    unchanged(ref_inputs, before)
    unchanged(res_inputs, before)
    assert not torch._C._is_alias_of(res_out, res_inputs["x"])

    x_fp32 = ref_inputs["x"].float()
    if not has_gate:
        # Contract: z=None is a plain RMSNorm times the affine weight.
        baseline = plain_norm(x_fp32, 1e-5, group_size) * ref_inputs["weight"].float()
        torch.testing.assert_close(ref_out.float(), baseline.to(dtype).float(),
                                   rtol=tolerance(dtype)[0], atol=tolerance(dtype)[1])
        ref_time = measure(lambda: kernelgenbench.baseline.v1_rmsnorm_gated_sglang(
            **ref_inputs, group_size=group_size))
        res_time = measure(lambda: kernelgenbench.solution.v1_rmsnorm_gated_sglang(
            **res_inputs, group_size=group_size))
        return result(ref_time, res_time)

    # Contract: gate multiplies x before the RMS statistic.  Compare against the
    # other order on the same data and require a visible difference.
    assert not torch._C._is_alias_of(res_out, res_inputs["gate"])
    z = F.silu(ref_inputs["gate"].float())
    gated = x_fp32 * z
    if group_size is None:
        gated_first = gated * torch.rsqrt(gated.square().mean(dim=-1, keepdim=True) + 1e-5)
        fused_first = x_fp32 * torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + 1e-5)
    else:
        gg = gated.reshape(*gated.shape[:-1], gated.shape[-1] // group_size, group_size)
        gated_first = (gg * torch.rsqrt(gg.square().mean(dim=-1, keepdim=True) + 1e-5)).reshape(
            gated.shape)
        xg = x_fp32.reshape(*x_fp32.shape[:-1], x_fp32.shape[-1] // group_size, group_size)
        fused_first = (xg * torch.rsqrt(xg.square().mean(dim=-1, keepdim=True) + 1e-5)).reshape(
            x_fp32.shape)
    weight = ref_inputs["weight"].float()
    expected = gated_first * weight
    other = fused_first * weight * z
    torch.testing.assert_close(ref_out.float(), expected.to(dtype).float(),
                               rtol=tolerance(dtype)[0], atol=tolerance(dtype)[1])
    assert not torch.allclose(expected, other, rtol=1e-2, atol=1e-2), \
        "gate must be applied before the RMS statistic"

    # Contract: the same entry point accepts z=None and then equals RMSNorm.
    plain = kernelgenbench.solution.v1_rmsnorm_gated_sglang(
        x=res_inputs["x"], gate=None, weight=res_inputs["weight"], eps=1e-5,
        group_size=group_size)
    torch.testing.assert_close(plain.float(),
                               (plain_norm(x_fp32, 1e-5, group_size)
                                * ref_inputs["weight"].float()).to(dtype).float(),
                               rtol=tolerance(dtype)[0], atol=tolerance(dtype)[1])

    ref_time = measure(lambda: kernelgenbench.baseline.v1_rmsnorm_gated_sglang(
        **ref_inputs, group_size=group_size))
    res_time = measure(lambda: kernelgenbench.solution.v1_rmsnorm_gated_sglang(
        **res_inputs, group_size=group_size))
    return result(ref_time, res_time)
