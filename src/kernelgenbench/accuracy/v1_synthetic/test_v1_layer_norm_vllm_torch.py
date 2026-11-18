"""Accuracy and performance for v1_layer_norm_vllm_torch (723 row 52).

Branch: old ``v1_layer_norm(..., backend="vllm")``.
Official entry: vLLM ``LayerNorm`` module; at the pinned commit ``forward_cuda``
delegates to ``forward_native`` (``F.layer_norm`` with promoted compute dtype
and ``type_as`` back to the input dtype).
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/layernorm.py#L221-L292

Inputs: ``vals`` is (batch..., hidden) and may be strided. ``gamma``/``beta`` may
be absent; when the parameter dtype is wider than the input the compute dtype is
promoted and the result is cast back to the input dtype. That ``None`` handling
and the input-dtype result are what separate this item from the DeepSpeed one.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic import rows52_61_checks as checks
from sandbox.verifier.test_parametrize import label, parametrize


# (shape, input dtype, parameter dtype or None, affine mode, layout)
COMBOS = [
    (shape, dtype, dtype, "gamma_beta", layout)
    for shape in [(2, 8, 32), (1, 4, 64), (2, 16, 128)]
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
    for layout in ("contiguous", "strided")
] + [
    (shape, torch.float32, None, "none", layout)
    for shape in [(2, 8, 32), (1, 4, 64)]
    for layout in ("contiguous", "strided")
] + [
    (shape, torch.bfloat16, torch.float32, "gamma_beta", layout)
    for shape in [(2, 8, 32), (1, 4, 64)]
    for layout in ("contiguous", "strided")
] + [
    ((2, 8, 32), torch.float16, torch.float32, "gamma_beta", "contiguous"),
    ((2, 16, 128), torch.bfloat16, torch.bfloat16, "gamma", "contiguous"),
    ((1, 4, 64), torch.float16, torch.float16, "gamma", "strided"),
]

TOLERANCE = {
    torch.float32: (4e-6, 2e-6),
    torch.float16: (2e-3, 1e-3),
    torch.bfloat16: (1e-2, 8e-3),
}


def input_build(config):
    shape, dtype, param_dtype, affine, layout = config
    generator = torch.Generator(device="cpu").manual_seed(20261115)
    vals = (torch.randn(shape, dtype=torch.float32, generator=generator) * 0.5).to(dtype)
    if layout == "strided":
        buffer = torch.zeros(shape[:-1] + (shape[-1] * 2,), dtype=dtype)
        buffer[..., ::2] = vals
        vals = buffer[..., ::2]
    gamma = beta = None
    if affine in ("gamma", "gamma_beta"):
        gamma = (torch.randn(shape[-1], dtype=torch.float32, generator=generator) * 0.2 + 1.0)
        gamma = gamma.to(param_dtype)
    if affine == "gamma_beta":
        beta = (torch.randn(shape[-1], dtype=torch.float32, generator=generator) * 0.1)
        beta = beta.to(param_dtype)
    return {"vals": vals, "gamma": gamma, "beta": beta, "epsilon": 1e-5}


def compute_ref(vals, gamma, beta, epsilon):
    return kernelgenbench.baseline.v1_layer_norm_vllm_torch(vals, gamma, beta, epsilon)


def compute_res(vals, gamma, beta, epsilon):
    return kernelgenbench.solution.v1_layer_norm_vllm_torch(vals, gamma, beta, epsilon)


def compare(ref_out, res_out, dtype):
    assert isinstance(ref_out, torch.Tensor) and isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == dtype, (
        f"vLLM LayerNorm returns the input dtype {dtype}, got {res_out.dtype}/{ref_out.dtype}")
    assert res_out.device == ref_out.device
    rtol, atol = TOLERANCE[dtype]
    torch.testing.assert_close(res_out.float(), ref_out.float(), rtol=rtol, atol=atol)


def compute_ref_performance(vals, gamma, beta, epsilon):
    return checks.measure(kernelgenbench.baseline.v1_layer_norm_vllm_torch,
                          {"vals": vals, "gamma": gamma, "beta": beta, "epsilon": epsilon},
                          kernelgenbench.device)


def compute_res_performance(vals, gamma, beta, epsilon):
    return checks.measure(kernelgenbench.solution.v1_layer_norm_vllm_torch,
                          {"vals": vals, "gamma": gamma, "beta": beta, "epsilon": epsilon},
                          kernelgenbench.device)


@label("v1_layer_norm_vllm_torch")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_layer_norm_vllm_torch(config):
    shape, dtype, param_dtype, affine, layout = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    for side in (ref_inputs, res_inputs):
        assert side["vals"].shape == shape
        assert side["vals"].stride() == inputs["vals"].stride()
        assert side["gamma"] is None if affine == "none" else side["gamma"] is not None
        assert side["beta"] is None if affine != "gamma_beta" else side["beta"] is not None

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    # Regardless of parameter dtype the output keeps the input dtype and is fresh.
    assert res_out.dtype == dtype
    for output, side in ((ref_out, ref_inputs), (res_out, res_inputs)):
        for name, value in side.items():
            if isinstance(value, torch.Tensor):
                assert not torch._C._is_alias_of(output, value), f"output aliases {name}"
                torch.testing.assert_close(value.cpu(), inputs[name], rtol=0, atol=0)
        if affine == "none":
            # No affine parameters at all: normalized rows have unit variance.
            variance = output.float().var(dim=-1, unbiased=False)
            torch.testing.assert_close(variance, torch.ones_like(variance), rtol=5e-3, atol=5e-3)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return checks.timings(ref_time, res_time)
