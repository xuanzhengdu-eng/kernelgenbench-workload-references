"""Accuracy and timing for DeepSpeed's zero-filling row gather.

src=(N, D); index=(M,) int32/int64 where entries outside [0, N) (e.g. -1 or
N) produce an all-zero row. Output=(M, D) with the source dtype, fresh storage.
``src`` and ``index`` are read-only; a non-contiguous ``src`` is accepted.
"""
import statistics
import time

import pytest
import torch
import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (source rows, feature width, index length, dtype, index dtype, strided source)
COMBOS = [
    (7, 8, 6, torch.float32, torch.int32, False),
    (7, 8, 6, torch.bfloat16, torch.int64, True),
    (7, 1032, 6, torch.int32, torch.int32, False),
    (0, 8, 6, torch.float32, torch.int32, False),
    (7, 8, 0, torch.float16, torch.int64, False),
    (5, 4, 6, torch.float16, torch.int64, True),
]


def input_build(config):
    rows, width, length, dtype, index_dtype, strided = config
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    backing = torch.randn((rows, width + (3 if strided else 0)), dtype=torch.float32,
                          generator=generator)
    if dtype is torch.int32:
        backing = backing.mul(16).to(torch.int32)
    else:
        backing = backing.to(dtype)
    src = backing[:, :width]
    if length:
        # Values include a duplicate, -1, an index == rows, and a too-large one.
        values = torch.tensor([4, -1, 1, 99, 0, 4][:length], dtype=index_dtype)
        values = torch.where(values < rows, values, torch.full_like(values, rows + 3))
    else:
        values = torch.empty((0,), dtype=index_dtype)
    return {"src": src, "index": values}


def fresh(inputs, device):
    return {
        name: value.to(device=device, copy=True) if isinstance(value, torch.Tensor) else value
        for name, value in inputs.items()
    }


def measure(fn, inputs):
    device = kernelgenbench.device
    samples = []
    for _ in range(6):
        arguments = fresh(inputs, device)
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn(**arguments)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        else:
            start = time.perf_counter()
            fn(**arguments)
            samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples[1:])


def compute_ref(src, index):
    return kernelgenbench.baseline.v1_gather_rows_zero_oob(src, index)


def compute_res(src, index):
    return kernelgenbench.solution.v1_gather_rows_zero_oob(src, index)


def expected_gather(src, index):
    rows = index.to(torch.int64)
    valid = (rows >= 0) & (rows < src.shape[0])
    expected = torch.zeros(
        (index.numel(), src.shape[1]), dtype=src.dtype, device=src.device
    )
    expected[valid] = src.contiguous()[rows[valid]]
    return expected


def compare(ref_out, res_out, config, ref_inputs, res_inputs, inputs):
    rows, width, length, dtype, index_dtype, strided = config
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape == (length, width)
    assert res_out.dtype == ref_out.dtype == dtype
    assert torch.equal(res_out, ref_out)
    expected = expected_gather(res_inputs["src"], res_inputs["index"])
    assert torch.equal(res_out, expected)
    # Out-of-range rows are exactly zero, not a wrapped read.
    oob = (res_inputs["index"] < 0) | (res_inputs["index"] >= rows)
    if oob.any():
        assert torch.count_nonzero(res_out[oob]) == 0
    assert not torch._C._is_alias_of(res_out, res_inputs["src"])
    for name in ("src", "index"):
        assert torch.equal(ref_inputs[name].cpu(), inputs[name]), f"reference changed {name}"
        assert torch.equal(res_inputs[name].cpu(), inputs[name]), f"solution changed {name}"
    return expected


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_gather_rows_zero_oob")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_gather_rows_zero_oob(config):
    rows, width, length, dtype, index_dtype, strided = config
    inputs = input_build(config)
    assert inputs["src"].shape == (rows, width)
    assert inputs["src"].is_contiguous() != strided
    assert inputs["index"].shape == (length,)
    assert inputs["index"].dtype == index_dtype

    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config, ref_inputs, res_inputs, inputs)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )
