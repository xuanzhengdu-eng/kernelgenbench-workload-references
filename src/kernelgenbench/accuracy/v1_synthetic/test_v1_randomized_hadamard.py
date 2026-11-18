"""Accuracy and performance for v1_randomized_hadamard.

Operator: vLLM ``single_rht(x, inverse)`` (randomized Hadamard transform used by
INT4 per-token-head quantization).

Official source (pinned commit):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/attention/ops/int4_per_token_head.py#L1132-L1167

Contract: fp32 ``x`` of shape ``(*, D)``; output has the same shape and is a
fresh tensor, the input is not modified.  ``D`` may be any positive integer: the
transform is applied block-diagonally with ``block = largest power-of-two
divisor of D``.  On this H800 the pinned dispatch takes the hadacore Tensor Core
kernel for every tested ``D``; hadacore round-trips through bfloat16, which is
why the value tolerance below is wider than fp32 matmul rounding (see the
reference docstring).

``compute_ref`` loads the transcribed exact-fp32 reference from
``kernelgenbench.baseline``; ``compute_res`` calls the operator bound as
``kernelgenbench.solution``, i.e. the real pinned ``single_rht`` during
verification.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows702_711_checks as checks


# Shapes include a power-of-two D, a non-power-of-two block split, a 3-D input and
# D values on both sides of the Triton tier's minimum of 16.
SHAPES = [(3, 8), (2, 24), (2, 3, 12), (4, 64), (2, 16), (2, 128)]
COMBOS = [
    (shape, inverse)
    for shape in SHAPES
    for inverse in (False, True)
]

# hadacore computes in bfloat16 internally (measured max |ref - op| = 0.104 for
# these shapes); the fp32 matmul path itself would be within a few ulp.  These
# bounds are the measured hadacore envelope with a small margin.
ATOL = 0.15
RTOL = 0.05


def input_build(config):
    """fp32 CPU tensor; shape is (*, D), with D the last dimension."""
    shape, inverse = config
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    return {
        "x": torch.randn(shape, dtype=torch.float32, generator=generator),
        "inverse": inverse,
    }


def compute_ref(x, inverse):
    return kernelgenbench.baseline.v1_randomized_hadamard(x, inverse)


def compute_res(x, inverse):
    return kernelgenbench.solution.v1_randomized_hadamard(x, inverse)


def compare(ref_out, res_out, dtype=None):
    """Same shape/dtype; values within the documented hadacore precision."""
    assert isinstance(res_out, torch.Tensor), type(res_out).__name__
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == torch.float32
    assert res_out.device == ref_out.device
    torch.testing.assert_close(res_out, ref_out, rtol=RTOL, atol=ATOL)


def compute_ref_performance(x, inverse):
    return checks.measure(
        kernelgenbench.baseline.v1_randomized_hadamard,
        {"x": x, "inverse": inverse},
        kernelgenbench.device,
    )


def compute_res_performance(x, inverse):
    return checks.measure(
        kernelgenbench.solution.v1_randomized_hadamard,
        {"x": x, "inverse": inverse},
        kernelgenbench.device,
    )


@label("v1_randomized_hadamard")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_randomized_hadamard(config):
    shape, inverse = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    assert res_inputs["x"].shape == shape

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, torch.float32)

    # A rotation is out-of-place: same shape, fresh storage, untouched input.
    block = shape[-1] & -shape[-1]
    assert res_out.shape == shape
    assert res_out.numel() == int(torch.tensor(shape).prod())
    assert block >= 1
    assert not torch._C._is_alias_of(res_out, res_inputs["x"])
    torch.testing.assert_close(res_inputs["x"].cpu(), inputs["x"], rtol=0, atol=0)

    return checks.timings(compute_ref_performance(**inputs),
                          compute_res_performance(**inputs))
