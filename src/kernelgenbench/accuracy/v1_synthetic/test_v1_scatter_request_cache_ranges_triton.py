"""Accuracy and performance for v1_scatter_request_cache_ranges_triton.

Row 542 of the original 723-item table, item OP-15541d7325c0, CUDA backend.  The
Solution is the real pinned Triton kernel ``assign_req_to_token_pool`` from
``sglang.srt.mem_cache.allocation``; the reference is the same contract written
with the identical ``@triton.jit`` body plus explicit argument checks.

Contract under test
-------------------
``req_to_token`` is a CUDA int32 ``[max_reqs, pool_len]`` table updated in
place, returned as ``None``.  ``req_pool_indices``, ``start_offset`` and
``end_offset`` live on the GPU (index dtypes int32/int64; both offsets share one
dtype) and ``out_cache_loc`` is int64 and read only.  The kernel builds the
exclusive prefix sum of ``end - start`` from the offsets of the earlier
programs, so:

``req_to_token[req, j] = int32(out_cache_loc[prefix + j - start])`` for
``j in [start, end)``.

Slots outside the ranges keep the sentinel ``-99``, the unused tail of
``out_cache_loc`` is never read, and int64 values outside the int32 range are
narrowed (``2147483648`` becomes ``-2147483648``).  The empty batch launches no
program and leaves the table alone.

COMBOS axes: (requests, starts, ends, cache locations), the index dtype for
``req_pool_indices`` and the dtype shared by the two offset arrays.  The table is
always ``[3, 8]`` int32 prefilled with ``-99``.
"""
import pytest
import torch

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (requests, starts, ends, cache locations); covers an unused input tail, an
# empty batch, a repeated request row, a zero range and a 64-bit value that
# does not fit in the int32 table.
CASES = [
    ([2, 0, 1], [0, 3, 5], [2, 3, 7], [9, -5, 14, 2147483648, 123]),
    ([], [], [], [3, 5]),
    ([1, 1], [0, 4], [2, 6], [1, 2, 3, 4]),
    ([0], [0], [0], []),
    ([2], [0], [8], list(range(100, 108))),
    ([0, 2, 1, 0], [0, 1, 2, 4], [1, 4, 4, 8], list(range(7, 17))),
]

COMBOS = [
    (case, index_dtype, offset_dtype)
    for case in CASES
    for index_dtype in (torch.int32, torch.int64)
    for offset_dtype in (torch.int32, torch.int64)
]

TABLE_SHAPE = (3, 8)
FILL = -99


def input_build(config, device=None):
    """CUDA inputs for one config (built on CPU, then moved)."""
    case, index_dtype, offset_dtype = config
    requests, starts, ends, locations = case
    device = device or kernelgenbench.device
    return {
        "req_pool_indices": torch.tensor(requests, dtype=index_dtype, device=device),
        "req_to_token": torch.full(TABLE_SHAPE, FILL, dtype=torch.int32, device=device),
        "start_offset": torch.tensor(starts, dtype=offset_dtype, device=device),
        "end_offset": torch.tensor(ends, dtype=offset_dtype, device=device),
        "out_cache_loc": torch.tensor(locations, dtype=torch.int64, device=device),
    }


def fresh(inputs):
    return {name: value.clone() for name, value in inputs.items()}


def _int32(value):
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value >= 0x80000000 else value


def oracle(inputs):
    """Expected CUDA table, built independently from the same inputs."""
    table = inputs["req_to_token"].clone()
    locations = inputs["out_cache_loc"].tolist()
    offset = 0
    for req, start, end in zip(inputs["req_pool_indices"].tolist(),
                               inputs["start_offset"].tolist(),
                               inputs["end_offset"].tolist()):
        for j in range(start, end):
            table[req, j] = _int32(locations[offset + j - start])
        offset += end - start
    return table


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_scatter_request_cache_ranges_triton(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_scatter_request_cache_ranges_triton(**inputs)


def compare(ref_out, res_out, config, ref_inputs, res_inputs, expected):
    """Return value, in-place table, input dtypes, sentinel slots and cast."""
    assert ref_out is None and res_out is None, "both sides return None"
    assert res_inputs["req_to_token"].dtype == torch.int32
    torch.cuda.synchronize()
    torch.testing.assert_close(ref_inputs["req_to_token"], expected, rtol=0, atol=0)
    torch.testing.assert_close(res_inputs["req_to_token"], expected, rtol=0, atol=0)
    for name in ("req_pool_indices", "start_offset", "end_offset", "out_cache_loc"):
        assert res_inputs[name].dtype == ref_inputs[name].dtype
        torch.testing.assert_close(res_inputs[name], ref_inputs[name], rtol=0, atol=0)
    locations = config[0][3]
    if any(value > 2147483647 for value in locations):
        assert (res_inputs["req_to_token"] < 0).any(), "int64 value was not narrowed"
    # Rows and columns that no range selects keep the sentinel.
    touched = torch.zeros(TABLE_SHAPE, dtype=torch.bool, device=res_inputs["req_to_token"].device)
    for req, start, end in zip(config[0][0], config[0][1], config[0][2]):
        touched[req, start:end] = True
    untouched = res_inputs["req_to_token"][~touched]
    assert torch.equal(untouched, torch.full_like(untouched, FILL))


def measure(fn, inputs, repeats=20):
    """Median single-call latency in milliseconds with CUDA events."""
    for _ in range(5):
        fn(**fresh(inputs))
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        arguments = fresh(inputs)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(**arguments)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


def compute_ref_performance(inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(inputs):
    return measure(compute_res, inputs)


@label("v1_scatter_request_cache_ranges_triton")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_scatter_request_cache_ranges_triton(config):
    inputs = input_build(config)
    ref_inputs = fresh(inputs)
    res_inputs = fresh(inputs)
    pointer = res_inputs["req_to_token"].data_ptr()
    expected = oracle(inputs)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    torch.cuda.synchronize()

    compare(ref_out, res_out, config, ref_inputs, res_inputs, expected)
    assert pointer == res_inputs["req_to_token"].data_ptr(), "table storage replaced"

    # The pinned Triton kernel validates nothing (the caller owns the capacity),
    # so only the reference's explicit check is exercised here; launching the
    # official kernel with a short buffer would read out of bounds.
    required = sum(end - start for start, end in zip(config[0][1], config[0][2]))
    if required:
        short_inputs = fresh(inputs)
        short_inputs["out_cache_loc"] = short_inputs["out_cache_loc"][: required - 1]
        with pytest.raises(AssertionError):
            compute_ref(**fresh(short_inputs))

    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
