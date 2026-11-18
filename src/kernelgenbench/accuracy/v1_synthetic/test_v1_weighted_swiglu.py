"""Accuracy and performance for v1_weighted_swiglu.

Operator: Megatron ``weighted_swiglu``.
Official source (pinned commit):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_bias_swiglu.py#L45-L48

Contract: ``y`` is (N, 2H) (a flattened (B, T, 2H) view), the first half is the
SiLU gate and the second half is the linear value; ``weights`` is (N, 1) and
broadcasts over the hidden axis.  The result is (N, H) and keeps ``y.dtype``.
The official helper is wrapped by ``@jit_fuser``.

Item: v1_weighted_swiglu (723-row 35).
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import activation_rows32_41_checks as checks


SHAPES = [(2, 16, 128), (2, 128, 1024)]
DTYPES = (torch.float32, torch.float16, torch.bfloat16)
COMBOS = [
    (shape, dtype, weight_dtype)
    for shape in SHAPES
    for dtype in DTYPES
    for weight_dtype in ((dtype,) if dtype == torch.float32 else (dtype, torch.float32))
]

TOLERANCE = {
    torch.float32: (4e-6, 2e-6),
    torch.float16: (3e-3, 2e-3),
    torch.bfloat16: (2e-2, 1e-2),
}


def input_build(config):
    """y: (B*T, 2H); weights: (B*T, 1)."""
    shape, dtype, weight_dtype = config
    batch, tokens, hidden = shape
    generator = torch.Generator(device="cpu").manual_seed(20261118)
    y = torch.randn((batch * tokens, 2 * hidden), dtype=dtype, generator=generator)
    weights = torch.rand((batch * tokens, 1), dtype=weight_dtype, generator=generator)
    return {"y": y, "weights": weights}


def compute_ref(y, weights):
    return kernelgenbench.baseline.v1_weighted_swiglu(y, weights)


def compute_res(y, weights):
    return kernelgenbench.solution.v1_weighted_swiglu(y, weights)


def compare(ref_out, res_out, dtype):
    assert isinstance(res_out, torch.Tensor), \
        f"expected a tensor, got {type(res_out).__name__}"
    assert res_out.shape == ref_out.shape, \
        f"shape {tuple(res_out.shape)} != {tuple(ref_out.shape)}"
    assert res_out.dtype == ref_out.dtype == dtype, \
        f"dtype {res_out.dtype} != {ref_out.dtype} != {dtype}"
    rtol, atol = TOLERANCE[dtype]
    torch.testing.assert_close(res_out.float(), ref_out.float(), rtol=rtol, atol=atol)


def compute_ref_performance(y, weights):
    return checks.measure(
        kernelgenbench.baseline.v1_weighted_swiglu,
        {"y": y, "weights": weights},
        kernelgenbench.device,
    )


def compute_res_performance(y, weights):
    return checks.measure(
        kernelgenbench.solution.v1_weighted_swiglu,
        {"y": y, "weights": weights},
        kernelgenbench.device,
    )


@label("v1_weighted_swiglu")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_weighted_swiglu(config):
    shape, dtype, weight_dtype = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    for side in (ref_inputs, res_inputs):
        assert side["y"].shape == (shape[0] * shape[1], 2 * shape[2])
        assert side["weights"].shape == (shape[0] * shape[1], 1)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype=dtype)

    for name, value in res_inputs.items():
        assert torch.equal(value, ref_inputs[name]), f"input {name} was modified"
    assert not torch._C._is_alias_of(res_out, res_inputs["y"]), \
        "output aliases y"
    assert not torch._C._is_alias_of(res_out, res_inputs["weights"]), \
        "output aliases weights"

    return checks.timings(compute_ref_performance(**inputs),
                          compute_res_performance(**inputs))
