"""Accuracy and performance for v1_geglu_back.

Operator: Megatron geglu_back (native (tanh)).
Official source (pinned commit):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_bias_geglu.py#L49-L66

Contract: chunks the last axis in two and concatenates the gradient halves; two inputs, no
  bias. Odd widths split unevenly via torch.chunk. Upstream runs under jit_fuser
  (=torch.compile), which fuses the pointwise chain and can differ from eager
  execution by an element ULP or two; the graded grid keeps the inputs small enough
  that both execution modes stay inside the project tolerance.

Item: v1_geglu_back (723-row 15). ``compute_ref`` loads this file's reference from
``kernelgenbench.baseline``; ``compute_res`` calls whatever the verifier bound as
``kernelgenbench.solution`` for the same label, i.e. the real operator of the
pinned commit during verification.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import activation_rows11_21_checks as checks


# Axes: (batch, seq, full_width) and dtype; the gradient g has full_width // 2 columns and
# y has full_width, both rounded once from fp32.
COMBOS = [
    ((2, 16, 128), torch.float32),
    ((2, 16, 128), torch.float16),
    ((2, 16, 128), torch.bfloat16),
    ((2, 128, 1024), torch.float32),
    ((2, 128, 1024), torch.float16),
    ((2, 128, 1024), torch.bfloat16),
    ((2, 128, 130), torch.float32),
    ((2, 128, 130), torch.float16),
    ((2, 128, 130), torch.bfloat16),
]


# Absolute and relative allowance per dtype: a couple of units in the last place for
# the fp32 accumulation order, one storage rounding for fp16/bf16.
TOLERANCE = {
    torch.float32: (4e-6, 2e-6),
    torch.float16: (2e-3, 1e-3),
    torch.bfloat16: (1e-2, 8e-3),
}


def input_build(config):
    """g: (batch, seq, width); y: (batch, seq, 2*width), so g's last axis is half of
    y's.
    """
    shape, dtype = config
    width = shape[-1] // 2
    generator = torch.Generator(device="cpu").manual_seed(20261115)
    g = torch.randn(shape[:-1] + (width,), dtype=dtype, generator=generator)
    y = torch.randn(shape, dtype=dtype, generator=generator)
    return {"g": (g.float() * 0.5).to(dtype), "y": (y.float() * 0.5).to(dtype)}


def compute_ref(g, y):
    """Official side: the reference loaded from ``kernelgenbench.baseline``."""
    return kernelgenbench.baseline.v1_geglu_back(g, y)


def compute_res(g, y):
    """Candidate side: whatever the verifier registered as ``kernelgenbench.solution``."""
    return kernelgenbench.solution.v1_geglu_back(g, y)


def compare(ref_out, res_out, dtype=None):
    """Both gradient halves concatenated in order: product half first, gate half
    second.
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


def compute_ref_performance(g, y):
    return checks.measure(
        kernelgenbench.baseline.v1_geglu_back,
        {"g": g, "y": y},
        kernelgenbench.device,
    )


def compute_res_performance(g, y):
    return checks.measure(
        kernelgenbench.solution.v1_geglu_back,
        {"g": g, "y": y},
        kernelgenbench.device,
    )


@label("v1_geglu_back")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_geglu_back(config):
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
