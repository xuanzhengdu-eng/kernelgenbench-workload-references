"""Accuracy and timing for SGLang all-layer KV slot copy.

The reference keeps the pinned CPU AOT kernel contract (gather every source row,
then scatter; overlapping slot lists are safe).  The solution binding calls the
official GPU Triton entry sglang.kernels.ops.kvcache.cache_move
.copy_all_layer_kv_cache_func, i.e. copy_all_layer_kv_cache_tiled on the H800.
The CPU AOT extension is not built in the pinned CUDA image, so that backend is
recorded as unverified; see the report row.  Destinations here are unique
because duplicate targets are only ordered by the CPU loop.
"""
import math
import statistics

import pytest
import torch

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (slots, per-buffer byte widths, source slots, destination slots, index dtype)
COMBOS = [
    (6, (8, 12), (0, 1, 2), (1, 2, 3), torch.int64),
    (5, (32, 16, 8), (0, 1, 4), (4, 0, 1), torch.int32),
    (4, (128,), (0, 1, 2), (3, 2, 0), torch.int64),
    (7, (256, 64), (2, 4, 6), (6, 2, 0), torch.int64),
    (4, (), (), (), torch.int64),
    (3, (6, 10), (), (), torch.int32),
    (8, (16, 24, 32, 40), (7, 0, 3), (0, 7, 5), torch.int64),
]


def input_build(config):
    slots, widths, src, dst, dtype = config
    buffers = [
        ((torch.arange(slots * width, dtype=torch.int32).reshape(slots, width) + 13 * i) % 251).to(torch.uint8)
        for i, width in enumerate(widths)
    ]
    return {
        "buffers": buffers,
        "tgt_loc": torch.tensor(dst, dtype=dtype),
        "src_loc": torch.tensor(src, dtype=dtype),
    }


def _to(inputs):
    device = kernelgenbench.device
    return {
        "buffers": [buf.to(device) for buf in inputs["buffers"]],
        "tgt_loc": inputs["tgt_loc"].to(device),
        "src_loc": inputs["src_loc"].to(device),
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_sgl_copy_all_layer_kv_(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_sgl_copy_all_layer_kv_(**inputs)


def compare(inputs, ref_inputs, res_inputs, config):
    slots, widths, src, dst, _ = config
    assert len(ref_inputs["buffers"]) == len(res_inputs["buffers"]) == len(widths)
    for i, (expected, actual) in enumerate(zip(ref_inputs["buffers"], res_inputs["buffers"])):
        assert actual.data_ptr() == res_inputs["buffers"][i].data_ptr()
        assert actual.shape == expected.shape == (slots, widths[i])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        untouched = set(range(slots)) - set(dst)
        for slot in untouched:
            torch.testing.assert_close(actual[slot], inputs["buffers"][i][slot].to(actual.device), rtol=0, atol=0)


def _bench(fn, inputs):
    samples = []
    for _ in range(6):
        args = _to(inputs)
        if args["buffers"] and args["buffers"][0].is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn(**args)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        else:
            import time
            begin = time.perf_counter()
            fn(**args)
            samples.append((time.perf_counter() - begin) * 1000.0)
    return statistics.median(samples[1:])


def compute_ref_performance(**inputs):
    return _bench(compute_ref, inputs)


def compute_res_performance(**inputs):
    return _bench(compute_res, inputs)


@label("v1_sgl_copy_all_layer_kv_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_copy_all_layer_kv_(config):
    inputs = input_build(config)
    ref_inputs = _to(inputs)
    res_inputs = _to(inputs)
    objects = list(res_inputs["buffers"])
    before = [buf.clone() for buf in res_inputs["buffers"]]
    assert compute_ref(**ref_inputs) is None
    assert compute_res(**res_inputs) is None
    for i, buf in enumerate(res_inputs["buffers"]):
        assert buf is objects[i], "buffer objects must be mutated in place"
    compare(inputs, ref_inputs, res_inputs, config)
    assert len(ref_inputs["buffers"]) == len(before)
    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time)
