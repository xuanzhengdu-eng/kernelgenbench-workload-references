"""Accuracy and performance for v1_weighted_swiglu_back.

Operator: Megatron weighted SwiGLU backward.
Official sources (pinned commit):

* helper: https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_bias_swiglu.py#L90-L97
* wrapper: https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_bias_swiglu.py#L191-L207

Contract: ``y`` is (B, T, 2H), ``g`` is (B, T, H), ``weights`` is (B, T, 1).
The direct helper returns ``(grad_y, grad_w)``.  ``full_backward=True`` selects
the upstream ``WeightedSwiGLUFunction.backward`` contract: it restores an FP8
saved ``y`` with ``original_dtype`` and returns ``(grad_y, grad_w, None)``.
The fork-only clamp / helper-clamped branch was removed because the fixed NVIDIA
commit has no clamped SwiGLU backward helper.

Item: v1_weighted_swiglu_back (723-row 34).
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import activation_rows32_41_checks as checks


SHAPES = [(2, 16, 128), (2, 128, 1024)]
DTYPES = (torch.float32, torch.float16, torch.bfloat16)
VARIANTS = ("helper", "wrapper", "wrapper_fp8")
COMBOS = [
    (shape, dtype, weight_dtype, variant)
    for shape in SHAPES
    for dtype in DTYPES
    for weight_dtype in ((dtype,) if dtype == torch.float32 else (dtype, torch.float32))
    for variant in VARIANTS
]

# Reduction-order allowance, measured on H800 against the jit_fuser output.
# grad_y is elementwise; grad_w sums H products in the low-precision dtype
# upstream, so it needs a wider absolute allowance.  Both remain far below the
# error produced by a swapped or missing gradient.
TOLERANCE_GRAD_Y = {
    torch.float32: (1e-5, 2e-6),
    torch.float16: (2e-2, 1e-2),
    torch.bfloat16: (5e-2, 8e-2),
}
TOLERANCE_GRAD_W = {
    torch.float32: (1e-5, 5e-6),
    torch.float16: (5e-2, 4e-2),
    torch.bfloat16: (5e-1, 3e-1),
}


def input_build(config):
    """g: (B, T, H); y: (B, T, 2H); weights: (B, T, 1)."""
    shape, dtype, weight_dtype, variant = config
    generator = torch.Generator(device="cpu").manual_seed(20261117)
    g = torch.randn(shape, dtype=dtype, generator=generator)
    y = torch.randn((*shape[:-1], 2 * shape[-1]), dtype=dtype, generator=generator)
    weights = torch.rand((*shape[:-1], 1), dtype=weight_dtype, generator=generator)
    inputs = {"g": g, "y": y, "weights": weights}
    if variant != "helper":
        inputs["full_backward"] = True
    if variant == "wrapper_fp8":
        inputs["y"] = y.to(torch.float8_e4m3fn)
        inputs["fp8_input_store"] = True
        inputs["original_dtype"] = dtype
    return inputs


def compute_ref(g, y, weights, full_backward=False, fp8_input_store=False,
                original_dtype=None):
    return kernelgenbench.baseline.v1_weighted_swiglu_back(
        g, y, weights, full_backward=full_backward,
        fp8_input_store=fp8_input_store, original_dtype=original_dtype)


def compute_res(g, y, weights, full_backward=False, fp8_input_store=False,
                original_dtype=None):
    return kernelgenbench.solution.v1_weighted_swiglu_back(
        g, y, weights, full_backward=full_backward,
        fp8_input_store=fp8_input_store, original_dtype=original_dtype)


def compare(ref_out, res_out, dtype, variant):
    expected_len = 2 if variant == "helper" else 3
    assert isinstance(ref_out, tuple) and len(ref_out) == expected_len, \
        f"reference returned {type(ref_out).__name__} of length " \
        f"{len(ref_out) if isinstance(ref_out, tuple) else '?'}, expected {expected_len}"
    assert isinstance(res_out, tuple) and len(res_out) == expected_len, \
        f"candidate returned {type(res_out).__name__} of length " \
        f"{len(res_out) if isinstance(res_out, tuple) else '?'}, expected {expected_len}"
    for index, (produced, expected) in enumerate(zip(res_out, ref_out)):
        if expected is None:
            assert produced is None, f"output {index} must be None"
            continue
        assert isinstance(produced, torch.Tensor), f"output {index} is not a tensor"
        assert produced.shape == expected.shape, \
            f"output {index} shape {tuple(produced.shape)} != {tuple(expected.shape)}"
        assert produced.dtype == expected.dtype, \
            f"output {index} dtype {produced.dtype} != {expected.dtype}"
        tolerance = TOLERANCE_GRAD_Y if index == 0 else TOLERANCE_GRAD_W
        rtol, atol = tolerance[dtype]
        torch.testing.assert_close(produced.float(), expected.float(),
                                   rtol=rtol, atol=atol)


def compute_ref_performance(**inputs):
    return checks.measure(
        kernelgenbench.baseline.v1_weighted_swiglu_back, inputs,
        kernelgenbench.device,
    )


def compute_res_performance(**inputs):
    return checks.measure(
        kernelgenbench.solution.v1_weighted_swiglu_back, inputs,
        kernelgenbench.device,
    )


@label("v1_weighted_swiglu_back")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_weighted_swiglu_back(config):
    shape, dtype, weight_dtype, variant = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    for side in (ref_inputs, res_inputs):
        assert side["g"].shape == shape
        assert side["y"].shape == (*shape[:-1], 2 * shape[-1])
        assert side["weights"].shape == (*shape[:-1], 1)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype=dtype, variant=variant)

    for name, value in res_inputs.items():
        if not isinstance(value, torch.Tensor):
            continue
        original = ref_inputs[name]
        if value.dtype == torch.float8_e4m3fn:
            unchanged = torch.equal(value.view(torch.uint8), original.view(torch.uint8))
        else:
            unchanged = torch.equal(value, original)
        assert unchanged, f"input {name} was modified"
    for produced in res_out:
        if isinstance(produced, torch.Tensor):
            assert not any(
                torch._C._is_alias_of(produced, value)
                for value in res_inputs.values()
                if isinstance(value, torch.Tensor)
            ), "an output aliases an input"

    return checks.timings(compute_ref_performance(**inputs),
                          compute_res_performance(**inputs))
