"""Accuracy and performance for v1_fast_gelu.

Operator: Megatron fast_gelu (native (tanh)).
Official source (pinned commit):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/activations.py#L20-L23

Contract: tanh approximation with constants 0.7978845608 and 0.044715 in that order. Bitwise
  equal to vLLM FastGELU and to DeepSpeed FPDT bias_gelu on all tested dtypes, so one
  item carries all three references.

Item: v1_fast_gelu (723-row 13). ``compute_ref`` loads this file's reference from
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
    return kernelgenbench.baseline.v1_fast_gelu(x)


def compute_res(x):
    """Candidate side: whatever the verifier registered as ``kernelgenbench.solution``."""
    return kernelgenbench.solution.v1_fast_gelu(x)


def compare(ref_out, res_out, dtype=None):
    """Tanh GELU with 0.7978845608; switching to exact erf moves fp16/bf16 by
    2.0e-3/1.6e-2.
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
        kernelgenbench.baseline.v1_fast_gelu,
        {"x": x},
        kernelgenbench.device,
    )


def compute_res_performance(x):
    return checks.measure(
        kernelgenbench.solution.v1_fast_gelu,
        {"x": x},
        kernelgenbench.device,
    )


@label("v1_fast_gelu")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_fast_gelu(config):
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

    return checks.timings(compute_ref_performance(**inputs),
                          compute_res_performance(**inputs))
