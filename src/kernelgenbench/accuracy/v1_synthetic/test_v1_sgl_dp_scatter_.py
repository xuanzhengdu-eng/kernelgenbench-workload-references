"""Accuracy and performance for v1_sgl_dp_scatter_.

The operator writes into a caller-allocated buffer, so each case checks the
preallocated destination's storage identity, its full contents (including the
zeroed tail), and that the gathered source is untouched.  Global rows carry
distinct values so a wrong offset or a missing zero fill cannot pass.
"""
import pytest
import torch
import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.bias_activation_test_utils import measure
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (global rows, local capacity, width, DP start, DP count, CP counts per rank,
#  CP rank, global DP counts, DP slot) -- the last two are only read in CP mode.
CONFIGS = [
    (8, 5, 16, 2, 3, None, 0, None, 0),
    (8, 0, 16, 0, 0, None, 0, None, 0),
    (10, 5, 8, 3, 4, [1, 2, 1], 1, [3, 4, 3], 1),
    (10, 4, 8, 0, 3, [0, 3], 0, [3, 4, 3], 0),
    (9, 3, 16, 6, 3, None, 0, None, 0),
    (12, 6, 8, 2, 4, [2, 2], 1, [5, 6], 1),
]
COMBOS = [(config, dtype) for config in CONFIGS
          for dtype in (torch.float32, torch.float16, torch.bfloat16)]


def input_build(config):
    (rows, local_rows, width, dp_start, dp_count, cp_counts, cp_rank,
     global_dp_counts, dp_slot), dtype = config
    # Distinct per-row values: global row r is filled with r + 1.
    rows_values = (torch.arange(rows, dtype=torch.float32)[:, None].expand(rows, width) + 1).to(dtype)
    # The destination starts at a sentinel so the zero fill is observable.
    local_tokens = torch.full((local_rows, width), -5.0, dtype=dtype)
    return {
        "local_tokens": local_tokens,
        "global_tokens": rows_values,
        "dp_start": dp_start,
        "dp_count": dp_count,
        "cp_counts": cp_counts,
        "cp_rank": cp_rank,
        "global_dp_counts": global_dp_counts,
        "dp_slot": dp_slot,
    }


def expected_local(inputs):
    local = torch.zeros_like(inputs["local_tokens"])
    if local.shape[0]:
        if inputs["cp_counts"] is None:
            start, count = inputs["dp_start"], inputs["dp_count"]
        else:
            start = (sum(inputs["global_dp_counts"][:inputs["dp_slot"]])
                     + sum(inputs["cp_counts"][:inputs["cp_rank"]]))
            count = inputs["cp_counts"][inputs["cp_rank"]]
        local[:count] = inputs["global_tokens"][start:start + count]
    return local


def compute_ref(local_tokens, global_tokens, dp_start, dp_count, cp_counts, cp_rank,
                global_dp_counts, dp_slot):
    return kernelgenbench.baseline.v1_sgl_dp_scatter_(
        local_tokens, global_tokens, dp_start, dp_count, cp_counts=cp_counts,
        cp_rank=cp_rank, global_dp_counts=global_dp_counts, dp_slot=dp_slot,
    )


def compute_res(local_tokens, global_tokens, dp_start, dp_count, cp_counts, cp_rank,
                global_dp_counts, dp_slot):
    return kernelgenbench.solution.v1_sgl_dp_scatter_(
        local_tokens, global_tokens, dp_start, dp_count, cp_counts=cp_counts,
        cp_rank=cp_rank, global_dp_counts=global_dp_counts, dp_slot=dp_slot,
    )


def compare(ref_out, res_out, ref_side, res_side, dtype, expected):
    # Both sides return None: the in-place write is the result.
    assert ref_out is None and res_out is None, \
        "the operator returns None and mutates its destination"
    for name, side in (("reference", ref_side), ("solution", res_side)):
        local = side["local_tokens"]
        assert local.dtype == dtype and local.device == expected.device
        assert local.is_contiguous(), f"{name}: destination must stay contiguous"
        assert torch.equal(local, expected), f"{name}: destination contents differ"
        assert torch.equal(side["global_tokens"], side["_global_before"]), \
            f"{name}: gathered source must not be modified"


def compute_ref_performance(local_tokens, global_tokens, dp_start, dp_count, cp_counts,
                            cp_rank, global_dp_counts, dp_slot):
    """Timed reference call; the destination is re-created per sample by measure()."""
    return measure(
        lambda **kw: compute_ref(**kw),
        {"local_tokens": local_tokens, "global_tokens": global_tokens,
         "dp_start": dp_start, "dp_count": dp_count, "cp_counts": cp_counts,
         "cp_rank": cp_rank, "global_dp_counts": global_dp_counts, "dp_slot": dp_slot},
    )


def compute_res_performance(local_tokens, global_tokens, dp_start, dp_count, cp_counts,
                            cp_rank, global_dp_counts, dp_slot):
    """Timed solution call under the same input conditions."""
    return measure(
        lambda **kw: compute_res(**kw),
        {"local_tokens": local_tokens, "global_tokens": global_tokens,
         "dp_start": dp_start, "dp_count": dp_count, "cp_counts": cp_counts,
         "cp_rank": cp_rank, "global_dp_counts": global_dp_counts, "dp_slot": dp_slot},
    )


@label("v1_sgl_dp_scatter_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_dp_scatter_(config):
    (rows, local_rows, width, dp_start, dp_count, cp_counts, cp_rank,
     global_dp_counts, dp_slot), dtype = config
    inputs = input_build(config)
    assert inputs["local_tokens"].shape == (local_rows, width)
    assert not torch._C._is_alias_of(inputs["local_tokens"], inputs["global_tokens"])

    def to_device():
        side = {
            name: value.clone() if isinstance(value, torch.Tensor) else value
            for name, value in inputs.items()
        }
        side = {name: value.to(kernelgenbench.device) if isinstance(value, torch.Tensor) else value
                for name, value in side.items()}
        side["_global_before"] = side["global_tokens"].clone()
        return side

    ref_side = to_device()
    res_side = to_device()
    expected = expected_local(inputs).to(kernelgenbench.device)
    ref_ptr = ref_side["local_tokens"].data_ptr()
    res_ptr = res_side["local_tokens"].data_ptr()
    ref_out = compute_ref(**{k: v for k, v in ref_side.items() if not k.startswith("_")})
    res_out = compute_res(**{k: v for k, v in res_side.items() if not k.startswith("_")})
    assert ref_side["local_tokens"].data_ptr() == ref_ptr, "reference replaced the destination storage"
    assert res_side["local_tokens"].data_ptr() == res_ptr, "solution replaced the destination storage"
    compare(ref_out, res_out, ref_side, res_side, dtype, expected)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
