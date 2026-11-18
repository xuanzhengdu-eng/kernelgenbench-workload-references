"""Accuracy and performance for v1_marlin_e8m0_scale_layout (vLLM MXFP8).

COMBOS cover short and odd-width scale tensors.  Inputs are exact positive
powers of two, so the E8M0 reinterpretation is lossless and byte equality is
the right check.  Group-of-four byte order is asserted against the upstream
``[0, 2, 1, 3]`` permutation, which distinguishes this item from the SGLang
FP4->FP8 variant that keeps the order and adds 6.
"""
import kernelgenbench
import pytest
import torch
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic.rows352_361_utils import (
    assert_unchanged,
    bytes_of,
    fresh,
    measure,
)

# (rows, columns); the column count is a multiple of four because the upstream
# permutation groups the flat scale bytes four at a time.
COMBOS = [
    (2, 8),
    (3, 16),
    (1, 4),
    (2, 12),
]


def input_build(config):
    rows, cols = config
    exponents = (torch.arange(rows * cols, dtype=torch.int32) % 20 + 118).reshape(rows, cols)
    scales = (exponents << 23).view(torch.float32)
    return {"marlin_scales": scales}


def compute_ref(marlin_scales):
    return kernelgenbench.baseline.v1_marlin_e8m0_scale_layout(marlin_scales)


def compute_res(marlin_scales):
    return kernelgenbench.solution.v1_marlin_e8m0_scale_layout(marlin_scales)


def compare(ref_out, res_out, dtype=None):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.dtype == torch.float8_e8m0fnu
    assert res_out.shape == ref_out.shape
    assert torch.equal(bytes_of(res_out), bytes_of(ref_out)), "E8M0 bytes differ"


def compute_ref_performance(marlin_scales):
    marlin_scales = marlin_scales.to(kernelgenbench.device)
    return measure(compute_ref, {"marlin_scales": marlin_scales},
                   reset=lambda: {"marlin_scales": marlin_scales.clone()})


def compute_res_performance(marlin_scales):
    marlin_scales = marlin_scales.to(kernelgenbench.device)
    return measure(compute_res, {"marlin_scales": marlin_scales},
                   reset=lambda: {"marlin_scales": marlin_scales.clone()})


@label("v1_marlin_e8m0_scale_layout")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_marlin_e8m0_scale_layout(config):
    rows, cols = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    before = res_inputs["marlin_scales"].clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out)

    # Upstream reorders each group of four exponent bytes as [0, 2, 1, 3].
    source = (before.view(torch.int32) >> 23).to(torch.uint8).reshape(-1).to(torch.int32)
    assert torch.equal(
        bytes_of(res_out).to(torch.int32),
        source.reshape(-1, 4)[:, [0, 2, 1, 3]].reshape(-1),
    ), "MXFP8 scale bytes are not in the upstream [0, 2, 1, 3] group order"
    assert not torch._C._is_alias_of(res_out, res_inputs["marlin_scales"])
    assert_unchanged({"marlin_scales": before}, res_inputs)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
