"""Accuracy and performance for v1_sgl_moe_cp_take_back.

``runtime`` is the padded CP gather: each rank's shard sits at
``rank * max(rows)`` and only ``rows[rank]`` of those rows are real.  The test
checks the returned tuple identity rules (residual passed through, the DP
buffer returned by identity, the untouched input returned by identity when there
is no slice and no DP scatter), the values, the zeroed DP tail, and that the
gathered input is never modified.
"""
import pytest
import torch
import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.bias_activation_test_utils import measure
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (padded rows, hidden width, real rows per CP rank or None, CP rank,
#  non-contiguous input, optional DP tuple (start, count, capacity))
CONFIGS = [
    (4, 16, None, 0, False, None),
    (9, 8, [2, 0, 3], 0, False, None),
    (9, 8, [2, 0, 3], 1, False, None),
    (9, 8, [2, 0, 3], 2, True, None),
    (8, 16, [4, 4], 1, False, None),
    (7, 8, None, 0, False, (2, 3, 5)),
    (9, 8, [2, 0, 3], 2, False, (1, 2, 4)),
]
COMBOS = [(config, dtype) for config in CONFIGS
          for dtype in (torch.float32, torch.float16, torch.bfloat16)]


def input_build(config):
    (padded, width, rows, cp_rank, strided, dp), dtype = config
    # Integral values keep the comparison exact for every dtype.
    hidden = torch.arange(padded * width, dtype=torch.float32).reshape(padded, width).to(dtype)
    if strided:
        storage = torch.full((padded, width * 2), -1.0, dtype=dtype)
        storage[:, ::2] = hidden
        hidden = storage[:, ::2]
    residual = torch.arange(2 * width, dtype=torch.float32).reshape(2, width).to(dtype)
    return {
        "hidden_states": hidden,
        "residual": residual,
        "rows": rows,
        "moe_cp_rank": cp_rank,
        "dp_output": None if dp is None else torch.full((dp[2], width), -3.0, dtype=dtype),
        "dp_start": 0 if dp is None else dp[0],
        "dp_count": 0 if dp is None else dp[1],
    }


def expected_output(inputs):
    hidden = inputs["hidden_states"]
    if inputs["rows"] is not None:
        rank = inputs["moe_cp_rank"]
        start = rank * max(inputs["rows"])
        hidden = hidden.narrow(0, start, inputs["rows"][rank]).contiguous()
    if inputs["dp_output"] is not None:
        buffer = torch.zeros_like(inputs["dp_output"])
        start, count = inputs["dp_start"], inputs["dp_count"]
        buffer[:count] = hidden[start:start + count]
        return buffer
    return hidden


def compute_ref(hidden_states, residual, rows, moe_cp_rank, dp_output, dp_start, dp_count):
    return kernelgenbench.baseline.v1_sgl_moe_cp_take_back(
        hidden_states, residual, rows, moe_cp_rank,
        dp_output=dp_output, dp_start=dp_start, dp_count=dp_count,
    )


def compute_res(hidden_states, residual, rows, moe_cp_rank, dp_output, dp_start, dp_count):
    return kernelgenbench.solution.v1_sgl_moe_cp_take_back(
        hidden_states, residual, rows, moe_cp_rank,
        dp_output=dp_output, dp_start=dp_start, dp_count=dp_count,
    )


def compare(ref_out, res_out, ref_side, res_side, config, dtype):
    (padded, width, rows, cp_rank, strided, dp), _ = config
    expected = expected_output(ref_side).to(kernelgenbench.device)
    for name, outputs, side in (("reference", ref_out, ref_side), ("solution", res_out, res_side)):
        assert isinstance(outputs, tuple) and len(outputs) == 2, f"{name}: expected a pair"
        hidden, residual = outputs
        assert isinstance(hidden, torch.Tensor) and isinstance(residual, torch.Tensor)
        assert residual is side["residual"], f"{name}: residual must be passed through by identity"
        assert hidden.dtype == dtype and hidden.device == expected.device
        assert hidden.shape == expected.shape, f"{name}: wrong row selection"
        assert hidden.is_contiguous(), f"{name}: result must be contiguous"
        assert torch.equal(hidden, expected), f"{name}: values differ"
        if side["dp_output"] is not None:
            assert hidden is side["dp_output"], f"{name}: must return the preallocated DP buffer"
        elif rows is None:
            assert hidden is side["hidden_states"], f"{name}: no slice means the input object is returned"
        torch.testing.assert_close(side["hidden_states"], side["_hidden_before"], rtol=0, atol=0)
        torch.testing.assert_close(side["residual"], side["_residual_before"], rtol=0, atol=0)
    # narrow().contiguous() keeps a contiguous slice a view of the gather, so the
    # two sides must agree on aliasing rather than both being copies.
    assert torch._C._is_alias_of(ref_out[0], ref_side["hidden_states"]) == \
        torch._C._is_alias_of(res_out[0], res_side["hidden_states"]), \
        "reference and solution disagree on gathered-input aliasing"


def compute_ref_performance(hidden_states, residual, rows, moe_cp_rank, dp_output,
                            dp_start, dp_count):
    return measure(compute_ref, {
        "hidden_states": hidden_states, "residual": residual, "rows": rows,
        "moe_cp_rank": moe_cp_rank, "dp_output": dp_output,
        "dp_start": dp_start, "dp_count": dp_count,
    })


def compute_res_performance(hidden_states, residual, rows, moe_cp_rank, dp_output,
                            dp_start, dp_count):
    return measure(compute_res, {
        "hidden_states": hidden_states, "residual": residual, "rows": rows,
        "moe_cp_rank": moe_cp_rank, "dp_output": dp_output,
        "dp_start": dp_start, "dp_count": dp_count,
    })


@label("v1_sgl_moe_cp_take_back")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_moe_cp_take_back(config):
    (padded, width, rows, cp_rank, strided, dp), dtype = config
    inputs = input_build(config)
    assert inputs["hidden_states"].shape == (padded, width)
    assert inputs["hidden_states"].is_contiguous() == (not strided)

    def fresh_side():
        side = {name: value.clone() if isinstance(value, torch.Tensor) else value
                for name, value in inputs.items()}
        side = {name: value.to(kernelgenbench.device) if isinstance(value, torch.Tensor) else value
                for name, value in side.items()}
        side["_hidden_before"] = side["hidden_states"].clone()
        side["_residual_before"] = side["residual"].clone()
        return side

    ref_side = fresh_side()
    res_side = fresh_side()
    ref_out = compute_ref(**{k: v for k, v in ref_side.items() if not k.startswith("_")})
    res_out = compute_res(**{k: v for k, v in res_side.items() if not k.startswith("_")})
    compare(ref_out, res_out, ref_side, res_side, (config[0], dtype), dtype)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
