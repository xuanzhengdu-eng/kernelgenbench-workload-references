"""Accuracy and performance for v1_geglu.

Operator: Megatron geglu (native (tanh)).
Official source (pinned commit):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_bias_geglu.py#L16-L27

Contract: torch.chunk split on the last axis, so an odd width splits into unequal halves and
  the two halves broadcast instead of intersecting: width 3 returns width 2, whereas
  vLLM/SGLang return width 1 with different values.

Item: v1_geglu (723-row 16). ``compute_ref`` loads this file's reference from
``kernelgenbench.baseline``; ``compute_res`` calls whatever the verifier bound as
``kernelgenbench.solution`` for the same label, i.e. the real operator of the
pinned commit during verification.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import activation_rows11_21_checks as checks


# Axes: input shape (even and odd last axis, plus the 1-column-per-half (3, 3) case),
# dtype, and layout (contiguous or last-axis stride 2).
COMBOS = [
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
    ((2, 128, 130), torch.float32, "contiguous"),
    ((2, 128, 130), torch.float32, "strided"),
    ((2, 128, 130), torch.float16, "contiguous"),
    ((2, 128, 130), torch.float16, "strided"),
    ((2, 128, 130), torch.bfloat16, "contiguous"),
    ((2, 128, 130), torch.bfloat16, "strided"),
    ((3, 3), torch.float32, "contiguous"),
    ((3, 3), torch.float32, "strided"),
    ((3, 3), torch.float16, "contiguous"),
    ((3, 3), torch.float16, "strided"),
    ((3, 3), torch.bfloat16, "contiguous"),
    ((3, 3), torch.bfloat16, "strided"),
]


# Absolute and relative allowance per dtype: a couple of units in the last place for
# the fp32 accumulation order, one storage rounding for fp16/bf16.
TOLERANCE = {
    torch.float32: (4e-6, 2e-6),
    torch.float16: (2e-3, 1e-3),
    torch.bfloat16: (1e-2, 8e-3),
}


def input_build(config):
    """y: one tensor of the requested shape/dtype whose last axis is the concatenated
    gate and value halves.
    """
    shape, dtype, layout = config
    generator = torch.Generator(device="cpu").manual_seed(20261115)
    values = torch.randn(shape, dtype=dtype, generator=generator)
    y = (values.float() * 0.5).to(dtype)
    if layout == "strided":
        buffer = torch.zeros(shape[:-1] + (shape[-1] * 2,), dtype=dtype)
        buffer[..., ::2] = y
        y = buffer[..., ::2]
    return {"y": y}


def compute_ref(y):
    """Official side: the reference loaded from ``kernelgenbench.baseline``."""
    return kernelgenbench.baseline.v1_geglu(y)


def compute_res(y):
    """Candidate side: whatever the verifier registered as ``kernelgenbench.solution``."""
    return kernelgenbench.solution.v1_geglu(y)


def compare(ref_out, res_out, dtype=None):
    """torch.chunk split with a tanh gate; an odd width keeps ceil(width/2) columns
    because the shorter second half broadcasts.
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


def compute_ref_performance(y):
    return checks.measure(
        kernelgenbench.baseline.v1_geglu,
        {"y": y},
        kernelgenbench.device,
    )


def compute_res_performance(y):
    return checks.measure(
        kernelgenbench.solution.v1_geglu,
        {"y": y},
        kernelgenbench.device,
    )


@label("v1_geglu")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_geglu(config):
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

    if config[0][-1] % 2:
        width = config[0][-1]
        assert ref_out.shape[-1] == width - width // 2, (
            f"torch.chunk splits {width} into {width - width // 2} + {width // 2} columns")

    return checks.timings(compute_ref_performance(**inputs),
                          compute_res_performance(**inputs))
