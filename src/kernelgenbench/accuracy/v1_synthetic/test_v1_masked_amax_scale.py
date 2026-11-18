"""Accuracy and timing for v1_masked_amax_scale.

Invalid rows carry a huge stale value (10000) and must not contribute to the
amax.  The three selector modes are row_valid, num_valid and per-expert
counts; quant_dtype selects the divisor (127 for int8, finfo.max for FP8).
The official implementation is a Triton kernel writing a one-element FP32
tensor, so the test compares that output and checks it stays float32.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows642_651_test_utils import (
    clone_tree,
    measure_ms,
    move_tree,
)
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


DTYPES = (torch.float32, torch.float16, torch.bfloat16)
MODES = ("mask", "prefix", "experts")
QUANT_DTYPES = (torch.int8, torch.float8_e4m3fn)

# (rows, cols, dtype, mode, quant_dtype, strided)
COMBOS = [
    (rows, cols, dtype, mode, quant_dtype, strided)
    for rows, cols in [(16, 31), (64, 129)]
    for dtype in DTYPES
    for mode in MODES
    for quant_dtype in QUANT_DTYPES
    for strided in ((False, True) if rows == 16 else (False,))
]


def input_build(config):
    rows, cols, dtype, mode, quant_dtype, strided = config
    generator = torch.Generator().manual_seed(650)
    raw = torch.randn(
        (rows, cols * (2 if strided else 1)),
        dtype=dtype,
        generator=generator,
    )
    x = raw[:, ::2] if strided else raw
    rows_index = torch.arange(rows)

    if mode == "mask":
        valid = rows_index % 3 == 0
        selectors = {"row_valid": valid}
    elif mode == "prefix":
        valid = rows_index < rows // 2
        selectors = {
            "num_valid": torch.tensor([rows // 2], dtype=torch.int32)
        }
    else:
        capacity = rows // 4
        counts = torch.tensor(
            [0, capacity, capacity // 2, 1], dtype=torch.int32
        )
        valid = (rows_index % capacity) < counts[rows_index // capacity]
        selectors = {
            "expert_num_tokens": counts,
            "rows_per_expert": capacity,
        }

    x = x.clone()
    x[~valid] = 10000.0
    return {
        "x": x,
        "quant_dtype": quant_dtype,
        "eps": 1e-6,
        "valid": valid,
        **selectors,
    }


def _split(inputs):
    arguments = {key: value for key, value in inputs.items() if key != "valid"}
    return arguments


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_masked_amax_scale(**_split(inputs))


def compute_res(**inputs):
    return kernelgenbench.solution.v1_masked_amax_scale(**_split(inputs))


def _expected(inputs):
    valid = inputs["valid"]
    scale_max = (
        127.0
        if inputs["quant_dtype"] == torch.int8
        else float(torch.finfo(inputs["quant_dtype"]).max)
    )
    values = torch.where(
        valid[:, None],
        inputs["x"].float().abs(),
        torch.zeros((), device=inputs["x"].device),
    )
    return (values.amax() / scale_max).clamp_min(inputs["eps"]).reshape(1)


def compare(ref_out, res_out, inputs):
    expected = _expected(inputs)
    assert isinstance(res_out, torch.Tensor)
    assert res_out.dtype == ref_out.dtype == torch.float32
    assert res_out.shape == ref_out.shape == (1,)
    assert res_out.device == inputs["x"].device
    torch.testing.assert_close(res_out, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(ref_out, expected, rtol=1e-6, atol=1e-7)


def compute_ref_performance(**inputs):
    return measure_ms(compute_ref, inputs, kernelgenbench.device)


def compute_res_performance(**inputs):
    return measure_ms(compute_res, inputs, kernelgenbench.device)


@label("v1_masked_amax_scale")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_masked_amax_scale(config):
    inputs = input_build(config)
    ref_inputs = move_tree(clone_tree(inputs), kernelgenbench.device)
    res_inputs = move_tree(clone_tree(inputs), kernelgenbench.device)
    ref_before = clone_tree(ref_inputs)
    res_before = clone_tree(res_inputs)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, ref_inputs)

    for name in ("x", "row_valid", "num_valid", "expert_num_tokens"):
        if name not in ref_inputs or not isinstance(
            ref_inputs[name], torch.Tensor
        ):
            continue
        torch.testing.assert_close(
            ref_inputs[name].cpu(), ref_before[name].cpu(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            res_inputs[name].cpu(), res_before[name].cpu(), rtol=0, atol=0
        )

    ref_time = compute_ref_performance(**ref_inputs)
    res_time = compute_res_performance(**res_inputs)
    return CustomBenchmarkResult(
        ref_time=ref_time,
        res_time=res_time,
        speedup=ref_time / res_time if res_time else float("inf"),
    )


def test_masked_amax_scale_requires_exactly_one_selector():
    x = torch.ones((4, 8), dtype=torch.float32)
    rows = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
    count = torch.tensor([2], dtype=torch.int32)
    with pytest.raises(AssertionError):
        compute_ref(x=x, quant_dtype=torch.int8, row_valid=rows, num_valid=count)
    with pytest.raises(AssertionError):
        compute_res(x=x, quant_dtype=torch.int8, row_valid=rows, num_valid=count)


def test_masked_amax_scale_requires_expert_pair():
    x = torch.ones((4, 8), dtype=torch.float32)
    counts = torch.tensor([2], dtype=torch.int32)
    with pytest.raises(AssertionError):
        compute_ref(
            x=x, quant_dtype=torch.int8, expert_num_tokens=counts
        )
    with pytest.raises(AssertionError):
        compute_res(
            x=x, quant_dtype=torch.int8, expert_num_tokens=counts
        )
