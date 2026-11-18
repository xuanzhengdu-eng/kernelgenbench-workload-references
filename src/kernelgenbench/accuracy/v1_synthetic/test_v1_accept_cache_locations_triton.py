"""Accuracy and performance for the CUDA/Triton branch of accept-cache-locations.

The pinned public entry is ``fill_accept_out_cache_loc_func``; on CUDA it runs
the Triton kernel ``fill_accept_out_cache_loc``.  The buffer contract is the
same as the CPU branch: gather ``out_cache_loc[accept_index]`` in flat order,
skip -1 (rejected) entries, write the compacted values to the front of the
caller-owned output, and leave the remaining capacity untouched.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.spec_decode_test_utils import fresh, measure
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# Synthetic accept vectors: no valid entries, repeats, a capacity tail and a
# larger step-shaped vector.
CASES = [
    ([-1, -2], [1, 2], 4),
    ([2, -1, 0, 2], [101, 102, 103], 6),
    ([], [], 0),
    ([0, 1], [8, 9], 2),
    ([3, -1, 7, 0, 7, 5, -1, 1] * 4, list(range(20)), 40),
]
COMBOS = [(case, dtype) for case in CASES for dtype in (torch.int32, torch.int64)]
PAD = -777


def input_build(config):
    case, dtype = config
    indices, values, capacity = case
    return {
        "accept_index": torch.tensor(indices, dtype=dtype),
        "out_cache_loc": torch.tensor(values, dtype=torch.int64),
        "accept_out_cache_loc": torch.full((capacity,), PAD, dtype=torch.int64),
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_accept_cache_locations_triton(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_accept_cache_locations_triton(**inputs)


def compact(case):
    """Independent expected compaction, including the untouched capacity tail."""
    indices, values, capacity = case
    kept = [values[index] for index in indices if index >= 0]
    return kept, [PAD] * (capacity - len(kept))


def compare(ref_out, res_out, dtype):
    assert ref_out is None and res_out is None


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs, device=kernelgenbench.device)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs, device=kernelgenbench.device)


@label("v1_accept_cache_locations_triton")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_accept_cache_locations_triton(config):
    case, dtype = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    assert ref_inputs["accept_index"].device.type == "cuda"
    assert ref_inputs["accept_index"].dtype == dtype
    assert ref_inputs["accept_index"].is_contiguous()
    assert ref_inputs["out_cache_loc"].dtype == torch.int64
    assert ref_inputs["accept_out_cache_loc"].numel() == case[2]

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    kept, tail = compact(case)
    for side in (ref_inputs, res_inputs):
        output = side["accept_out_cache_loc"]
        assert output.dtype == torch.int64
        assert output[:len(kept)].tolist() == kept
        assert output[len(kept):].tolist() == tail, "capacity tail must stay untouched"
    # The source vectors and the output buffer are distinct objects, and the
    # index/value inputs are read-only.
    assert not torch._C._is_alias_of(ref_inputs["accept_out_cache_loc"],
                                     ref_inputs["out_cache_loc"])
    for name in ("accept_index", "out_cache_loc"):
        assert torch.equal(ref_inputs[name], inputs[name].to(ref_inputs[name].device))

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
