"""Accuracy and performance for v1_greedy_argmax.

Every row stores 10.0 at token 0 and at the last token, so a correct argmax ties
to index 0. Row 0 of the NaN cases carries a NaN at token 1: the split kernel
treats it as -inf (token 0 wins), while torch.argmax returns the NaN index. That
difference is asserted explicitly per backend path.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.sampling_test_utils import (
    assert_unchanged,
    measure_call,
    move,
)
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

SHAPES = [(0, 4097), (2, 64), (2, 4096), (2, 4097), (2, 8192)]
COMBOS = [
    (shape, dtype, strided, nan)
    for shape in SHAPES
    for dtype in (torch.float32, torch.bfloat16, torch.float16)
    for strided in (False, True)
    for nan in (False, True)
    if not (nan and shape[0] == 0)
]


def input_build(config):
    (rows, vocab), dtype, strided, nan = config
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    # Small magnitudes keep the two 10.0 anchors the unique maximum.
    base = (
        (torch.randn((rows, vocab * (2 if strided else 1)), generator=generator) * 0.5)
        .clamp(-1.0, 1.0)
        .to(dtype)
    )
    logits = base[:, ::2] if strided else base
    if rows:
        # Write through the view so the strided case keeps stride(1) == 2.
        logits[:, 0] = 10.0
        logits[:, vocab - 1] = 10.0
        if nan:
            logits[0, 1] = float("nan")
    return {"logits": logits}


def compute_ref(logits):
    return kernelgenbench.baseline.v1_greedy_argmax(logits)


def compute_res(logits):
    return kernelgenbench.solution.v1_greedy_argmax(logits)


def split_path(logits):
    """The exact condition the official entry uses to pick its Triton kernel."""
    return (
        logits.is_cuda
        and logits.dtype in (torch.bfloat16, torch.float32)
        and logits.ndim == 2
        and logits.stride(1) == 1
        and logits.shape[1] >= 4096
    )


def compare(ref_out, res_out, dtype):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.dtype == torch.int64
    assert res_out.shape == ref_out.shape
    assert res_out.device == ref_out.device
    assert torch.equal(res_out, ref_out)


def compute_ref_performance(logits):
    return measure_call(compute_ref, {"logits": logits}, logits.device)


def compute_res_performance(logits):
    return measure_call(compute_res, {"logits": logits}, logits.device)


@label("v1_greedy_argmax")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_greedy_argmax(config):
    shape, dtype, strided, nan = config
    inputs = input_build(config)
    device = kernelgenbench.device
    ref_inputs = move(inputs, device)
    res_inputs = move(inputs, device)
    before = res_inputs["logits"].clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)
    assert_unchanged({"logits": res_inputs["logits"]}, {"logits": before})

    if shape[0]:
        # Ties resolve to the earliest index.
        assert torch.equal(res_out[1:] if nan else res_out, torch.zeros_like(res_out[1:] if nan else res_out))
        if nan:
            expected = 0 if split_path(res_inputs["logits"]) else 1
            assert int(res_out[0]) == expected, (dtype, strided, expected, res_out)
            plain = torch.argmax(res_inputs["logits"].to(torch.float32), dim=-1)
            if split_path(res_inputs["logits"]):
                # The split kernel's NaN rule differs from torch.argmax.
                assert int(plain[0]) == 1

    perf = move(inputs, device)
    ref_time = compute_ref_performance(**perf)
    res_time = compute_res_performance(**perf)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )
