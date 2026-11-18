"""Accuracy and performance for v1_squared_relu.

Operator: Megatron squared_relu (native (pow)).
Official source (pinned commit):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/activations.py#L8-L11

Contract: pow(relu(x), 2). Forward is bitwise equal to vLLM's square(relu) and SGLang's
  multiply form. The three share one item: measured with the pinned operators, the
  only divergence is the fp16 input gradient at an overflowing product (x=40000,
  upstream 0.1 gives 8000 for the jit-fused Megatron operator and for SGLang's
  multiply, but inf for vLLM's eager square), which depends on the autograd
  decomposition of the execution engine rather than on the operator, so the graded
  grid uses finite gradients and the edge case is recorded as evidence instead of
  being asserted.

Item: v1_squared_relu (723-row 21). ``compute_ref`` loads this file's reference from
``kernelgenbench.baseline``; ``compute_res`` calls whatever the verifier bound as
``kernelgenbench.solution`` for the same label, i.e. the real operator of the
pinned commit during verification.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import activation_rows11_21_checks as checks


# Axes: input shape (including the empty (0, 13) case), dtype, and layout. A "strided"
# entry keeps the same values and shape but puts them at stride 2 of a wider buffer, so
# the tensor is non-contiguous; single-element axes skip that entry.
COMBOS = [
    ((0, 13), torch.float32, "contiguous"),
    ((0, 13), torch.float16, "contiguous"),
    ((0, 13), torch.bfloat16, "contiguous"),
    ((3, 7), torch.float32, "contiguous"),
    ((3, 7), torch.float32, "strided"),
    ((3, 7), torch.float16, "contiguous"),
    ((3, 7), torch.float16, "strided"),
    ((3, 7), torch.bfloat16, "contiguous"),
    ((3, 7), torch.bfloat16, "strided"),
    ((2, 16, 128), torch.float32, "contiguous"),
    ((2, 16, 128), torch.float32, "strided"),
    ((2, 16, 128), torch.float16, "contiguous"),
    ((2, 16, 128), torch.float16, "strided"),
    ((2, 16, 128), torch.bfloat16, "contiguous"),
    ((2, 16, 128), torch.bfloat16, "strided"),
    ((2, 128, 1024), torch.float32, "contiguous"),
    ((2, 128, 1024), torch.float32, "strided"),
    ((2, 128, 1024), torch.float16, "contiguous"),
    ((2, 128, 1024), torch.float16, "strided"),
    ((2, 128, 1024), torch.bfloat16, "contiguous"),
    ((2, 128, 1024), torch.bfloat16, "strided"),
]


# Absolute and relative allowance per dtype: a couple of units in the last place for
# the fp32 accumulation order, one storage rounding for fp16/bf16.
TOLERANCE = {
    torch.float32: (4e-6, 2e-6),
    torch.float16: (2e-3, 1e-3),
    torch.bfloat16: (1e-2, 8e-3),
}


def input_build(config):
    """x: one tensor of the requested shape/dtype; the strided variant is stride 2.
    """
    shape, dtype, layout = config
    generator = torch.Generator(device="cpu").manual_seed(20261115)
    x = torch.randn(shape, dtype=dtype, generator=generator)
    if layout == "strided":
        buffer = torch.zeros(shape[:-1] + (shape[-1] * 2,), dtype=dtype)
        buffer[..., ::2] = x
        x = buffer[..., ::2]
    return {"x": x}


def compute_ref(x):
    """Official side: the reference loaded from ``kernelgenbench.baseline``."""
    return kernelgenbench.baseline.v1_squared_relu(x)


def compute_res(x):
    """Candidate side: whatever the verifier registered as ``kernelgenbench.solution``."""
    return kernelgenbench.solution.v1_squared_relu(x)


def compare(ref_out, res_out, dtype=None):
    """square of the rectified input, plus a matching fp16 input gradient.
    """
    assert isinstance(res_out, torch.Tensor), \
        f"expected a tensor, got {type(res_out).__name__}"
    assert res_out.shape == ref_out.shape, \
        f"shape {tuple(res_out.shape)} != {tuple(ref_out.shape)}"
    assert res_out.dtype == ref_out.dtype, \
        f"dtype {res_out.dtype} != {ref_out.dtype}"
    if ref_out.numel() == 0:
        return
    rtol, atol = TOLERANCE[dtype or ref_out.dtype]
    torch.testing.assert_close(res_out.float(), ref_out.float(), rtol=rtol, atol=atol,
                               equal_nan=True)


def compute_ref_performance(x):
    return checks.measure(
        kernelgenbench.baseline.v1_squared_relu,
        {"x": x},
        kernelgenbench.device,
    )


def compute_res_performance(x):
    return checks.measure(
        kernelgenbench.solution.v1_squared_relu,
        {"x": x},
        kernelgenbench.device,
    )


@label("v1_squared_relu")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_squared_relu(config):
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype=checks.config_dtype(config))

    # Side effects: neither side may modify its inputs, and the result must be fresh.
    for key, value in res_inputs.items():
        if isinstance(value, torch.Tensor) and value.numel():
            assert torch.equal(value, ref_inputs[key]), f"input {key} was modified"
    for produced in (res_out if isinstance(res_out, tuple) else (res_out,)):
        assert not any(
            torch._C._is_alias_of(produced, value)
            for value in res_inputs.values()
            if isinstance(value, torch.Tensor) and value.numel()
        ), "output aliases an input"

    # fp16 input gradient on finite inputs: the pow, square and multiply forms must
    # agree exactly, which is what keeps this item from splitting on gradients.
    grad_x = torch.tensor([100.0, -3.0, 1e-3, 0.0, 0.25], dtype=torch.float16,
                          device=kernelgenbench.device)
    upstream = torch.tensor([0.1, 0.5, 0.1, 0.1, 0.1], dtype=torch.float16,
                            device=kernelgenbench.device)
    ref_x = grad_x.clone().requires_grad_()
    res_x = grad_x.clone().requires_grad_()
    compute_ref(ref_x).backward(upstream)
    compute_res(res_x).backward(upstream)
    assert res_x.grad.dtype == ref_x.grad.dtype, "gradient dtype changed"
    torch.testing.assert_close(res_x.grad.float(), ref_x.grad.float(), rtol=0, atol=0,
                               equal_nan=True)

    return checks.timings(compute_ref_performance(**inputs),
                          compute_res_performance(**inputs))
