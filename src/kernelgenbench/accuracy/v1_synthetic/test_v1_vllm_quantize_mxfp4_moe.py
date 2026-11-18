"""Accuracy and performance for v1_vllm_quantize_mxfp4_moe.

COMBOS cover float16/bfloat16 expert weights, all-zero blocks, exact E2M1
tie values, zero rows expanded by padding, wide magnitudes and non-power-of-two
token counts.  The test checks both packed nibbles and scale bytes exactly,
plus shape/dtype/contiguity and the no-write contract.
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

# (experts, rows, K, dtype, pattern)
COMBOS = [
    (1, 1, 32, torch.float16, "zero"),
    (1, 3, 32, torch.float16, "ties"),
    (2, 4, 64, torch.bfloat16, "mixed"),
    (3, 2, 96, torch.float16, "mixed"),
    (2, 5, 128, torch.bfloat16, "zero_rows"),
    (4, 8, 64, torch.float16, "wide"),
]


def input_build(config):
    experts, rows, columns, dtype, pattern = config
    values = torch.arange(
        experts * rows * columns, dtype=torch.float32
    ).reshape(experts, rows, columns)
    values = (((values * 5 + 3) % 31) - 15) * 0.125
    if pattern == "zero":
        values.zero_()
    elif pattern == "ties":
        levels = torch.tensor(
            [0.0, -0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5,
             1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0],
            dtype=torch.float32,
        )
        values = levels.repeat(experts * rows * columns // levels.numel()).reshape(
            experts, rows, columns
        )
    elif pattern == "zero_rows":
        values[::2].zero_()
    elif pattern == "wide":
        values = values * 512
    return {"weight": values.to(dtype)}


def compute_ref(weight):
    return kernelgenbench.baseline.v1_vllm_quantize_mxfp4_moe(weight)


def compute_res(weight):
    return kernelgenbench.solution.v1_vllm_quantize_mxfp4_moe(weight)


def compare(ref_out, res_out, dtype=None):
    assert isinstance(res_out, tuple) and len(res_out) == 2
    for index, (expected, actual) in enumerate(zip(ref_out, res_out)):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == torch.uint8
        assert actual.shape == expected.shape
        assert actual.is_contiguous()
        assert torch.equal(bytes_of(actual), bytes_of(expected)), f"output {index} bytes"


def _args(weight):
    def build():
        return {"weight": weight.clone()}
    return build


def compute_ref_performance(weight):
    return measure(compute_ref, None, reset=_args(weight.to(kernelgenbench.device)))


def compute_res_performance(weight):
    return measure(compute_res, None, reset=_args(weight.to(kernelgenbench.device)))


@label("v1_vllm_quantize_mxfp4_moe")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_quantize_mxfp4_moe(config):
    experts, rows, columns, dtype, pattern = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    before = res_inputs["weight"].clone()
    assert res_inputs["weight"].shape == (experts, rows, columns)
    assert res_inputs["weight"].dtype == dtype

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out)

    packed, scales = res_out
    assert packed.shape == (experts, rows, columns // 2)
    assert scales.shape == (experts, rows, columns // 32)
    assert not any(torch._C._is_alias_of(output, res_inputs["weight"])
                   for output in res_out)
    assert_unchanged({"weight": before}, res_inputs)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
