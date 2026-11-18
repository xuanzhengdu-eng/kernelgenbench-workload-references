"""Accuracy and performance for v1_swiglu_oai.

Operator: vLLM ``SwigluOAIAndMul`` (GPT-OSS interleaved SwiGLU).
Official source (pinned commit):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/activation.py#L486-L493

Contract: last dimension ``2H`` with interleaved ``(gate, up)`` pairs.
``gate = min(x[..., ::2], limit)``, ``up = clamp(x[..., 1::2], -limit, limit)``,
output ``(up + 1) * gate * sigmoid(alpha * gate)`` of shape
``x.shape[:-1] + (H,)`` and the input dtype.  The upstream CUDA branch calls
``torch.ops._C.swigluoai_and_mul``; the verifier binds the real module.

This is a distinct Operator Item from SGLang's Phi3 ``gegelu`` because the
latter preserves infinities and uses ``reshape(-1, 2)`` pair order.

Item: v1_swiglu_oai (vLLM, 723-row 40).
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import activation_rows32_41_checks as checks


SHAPES = [(2, 16, 128), (2, 128, 1024)]
DTYPES = (torch.float32, torch.float16, torch.bfloat16)
LIMITS = (1.5, 7.0)
ALPHA = 1.702
COMBOS = [(shape, dtype, limit) for shape in SHAPES for dtype in DTYPES
          for limit in LIMITS]

TOLERANCE = {
    torch.float32: (4e-6, 2e-6),
    torch.float16: (3e-3, 2e-3),
    torch.bfloat16: (2e-2, 1e-2),
}


def input_build(config):
    """x: (..., 2H); values deliberately straddle the symmetric/asymmetric clamps."""
    shape, dtype, limit = config
    generator = torch.Generator(device="cpu").manual_seed(20261122)
    x = torch.randn((*shape[:-1], 2 * shape[-1]), dtype=dtype, generator=generator) * 3.0
    flat = x.view(-1)
    flat[0] = 10.0
    flat[1] = -10.0
    flat[2] = limit + 1.0
    flat[3] = -(limit + 1.0)
    return {"x": x, "alpha": ALPHA, "limit": limit}


def compute_ref(x, alpha, limit):
    return kernelgenbench.baseline.v1_swiglu_oai(x, alpha, limit)


def compute_res(x, alpha, limit):
    return kernelgenbench.solution.v1_swiglu_oai(x, alpha, limit)


def compare(ref_out, res_out, dtype, limit):
    assert isinstance(res_out, torch.Tensor), \
        f"expected a tensor, got {type(res_out).__name__}"
    assert res_out.shape == ref_out.shape, \
        f"shape {tuple(res_out.shape)} != {tuple(ref_out.shape)}"
    assert res_out.dtype == ref_out.dtype == dtype, \
        f"dtype {res_out.dtype} != {ref_out.dtype} != {dtype}"
    rtol, atol = TOLERANCE[dtype]
    torch.testing.assert_close(res_out.float(), ref_out.float(), rtol=rtol, atol=atol)


def compute_ref_performance(**inputs):
    return checks.measure(
        kernelgenbench.baseline.v1_swiglu_oai, inputs, kernelgenbench.device)


def compute_res_performance(**inputs):
    return checks.measure(
        kernelgenbench.solution.v1_swiglu_oai, inputs, kernelgenbench.device)


@label("v1_swiglu_oai")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_swiglu_oai(config):
    shape, dtype, limit = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    for side in (ref_inputs, res_inputs):
        assert side["x"].shape == (*shape[:-1], 2 * shape[-1])
        assert side["alpha"] == ALPHA and side["limit"] == limit

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype=dtype, limit=limit)

    assert res_out.shape == (*shape[:-1], shape[-1])
    assert torch.equal(res_inputs["x"], ref_inputs["x"]), "input x was modified"
    assert not torch._C._is_alias_of(res_out, res_inputs["x"]), \
        "output aliases x"

    return checks.timings(compute_ref_performance(**inputs),
                          compute_res_performance(**inputs))
