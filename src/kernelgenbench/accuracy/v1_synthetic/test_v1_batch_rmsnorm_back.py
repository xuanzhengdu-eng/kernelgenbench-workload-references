"""Accuracy and performance for v1_batch_rmsnorm_back (723 row 61).

Official entry: Megatron ``BatchInvariantRMSNormFn.backward`` with
``zero_centered_gamma=False`` and the forward's saved ``(x, weight, rsigma)``.
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/custom_layers/batch_invariant_kernels.py#L912-L935

Inputs: ``grad_output`` and ``x`` are (batch..., hidden) with the same shape,
``weight`` is (hidden,), ``rsigma`` is the fp32 saved ``rsqrt(mean(x^2)+eps)``
with shape (batch..., 1). Key assertions: the result is the upstream four-tuple
``(dx, g_w, None, None)``, ``dx`` keeps ``x.dtype`` and ``g_w`` keeps
``weight.dtype``, the fp32 formula is reconstructed independently (including the
extra weight factor in the second dx term), and no input is modified.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic import rows52_61_checks as checks
from sandbox.verifier.test_parametrize import label, parametrize


COMBOS = [
    (shape, dtype, layout)
    for shape in [(2, 16, 128), (2, 128, 1024), (3, 7, 64)]
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
    for layout in ("contiguous", "strided")
]

TOLERANCE = {
    torch.float32: (2e-6, 1e-6),
    torch.float16: (2e-3, 1e-3),
    torch.bfloat16: (1e-2, 8e-3),
}


def input_build(config):
    shape, dtype, layout = config
    generator = torch.Generator(device="cpu").manual_seed(20261115)
    x = (torch.randn(shape, generator=generator, dtype=torch.float32)).to(dtype)
    grad_output = (torch.randn(shape, generator=generator, dtype=torch.float32) * 0.5).to(dtype)
    if layout == "strided":
        # Last axis stride 2; both operands keep the same layout relation.
        def widen(t):
            buffer = torch.zeros(shape[:-1] + (shape[-1] * 2,), dtype=dtype)
            buffer[..., ::2] = t
            return buffer[..., ::2]
        x, grad_output = widen(x), widen(grad_output)
    weight = torch.randn(shape[-1], generator=generator, dtype=torch.float32).to(dtype)
    rsigma = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
    return {"grad_output": grad_output, "x": x, "weight": weight, "rsigma": rsigma}


def compute_ref(grad_output, x, weight, rsigma):
    return kernelgenbench.baseline.v1_batch_rmsnorm_back(grad_output, x, weight, rsigma)


def compute_res(grad_output, x, weight, rsigma):
    return kernelgenbench.solution.v1_batch_rmsnorm_back(grad_output, x, weight, rsigma)


def compare(ref_out, res_out, dtype, weight_dtype):
    assert isinstance(ref_out, tuple) and len(ref_out) == 4, "reference must return 4 items"
    assert isinstance(res_out, tuple) and len(res_out) == 4, "solution must return 4 items"
    assert res_out[2] is None and res_out[3] is None, "no grad for eps/zero_centered_gamma"
    assert isinstance(res_out[0], torch.Tensor) and res_out[0].shape == ref_out[0].shape
    assert res_out[0].dtype == ref_out[0].dtype == dtype
    assert isinstance(res_out[1], torch.Tensor) and res_out[1].shape == ref_out[1].shape
    assert res_out[1].dtype == ref_out[1].dtype == weight_dtype
    rtol, atol = TOLERANCE[dtype]
    torch.testing.assert_close(res_out[0].float(), ref_out[0].float(), rtol=rtol, atol=atol)
    torch.testing.assert_close(res_out[1].float(), ref_out[1].float(), rtol=rtol, atol=atol)


def compute_ref_performance(**inputs):
    return checks.measure(kernelgenbench.baseline.v1_batch_rmsnorm_back, inputs,
                          kernelgenbench.device)


def compute_res_performance(**inputs):
    return checks.measure(kernelgenbench.solution.v1_batch_rmsnorm_back, inputs,
                          kernelgenbench.device)


@label("v1_batch_rmsnorm_back")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_batch_rmsnorm_back(config):
    shape, dtype, layout = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    assert res_inputs["x"].shape == shape and res_inputs["grad_output"].shape == shape
    assert res_inputs["rsigma"].shape == (*shape[:-1], 1)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype, inputs["weight"].dtype)

    # Independent reconstruction of the source formula, including the extra
    # weight factor in the second dx term.
    g = res_inputs["grad_output"].float()
    v = res_inputs["x"].float()
    w = res_inputs["weight"].float()
    r = res_inputs["rsigma"]
    s = (g * v * w).sum(-1, keepdim=True)
    dx = g * (w * r) - (w * r * r * r) * (s * v) / shape[-1]
    torch.testing.assert_close(res_out[0].float(), dx.to(dtype).float(),
                               rtol=TOLERANCE[dtype][0], atol=TOLERANCE[dtype][1])
    torch.testing.assert_close(
        res_out[1].float(),
        (g * v * r).sum(dim=tuple(range(g.ndim - 1))).to(inputs["weight"].dtype).float(),
        rtol=TOLERANCE[dtype][0], atol=TOLERANCE[dtype][1])

    for name, value in res_inputs.items():
        assert not torch._C._is_alias_of(res_out[0], value), f"dx aliases {name}"
        torch.testing.assert_close(value.cpu(), inputs[name], rtol=0, atol=0)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return checks.timings(ref_time, res_time)
