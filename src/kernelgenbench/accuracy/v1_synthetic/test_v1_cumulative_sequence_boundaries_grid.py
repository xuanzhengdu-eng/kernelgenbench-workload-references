"""Accuracy and CPU timing for SGLang cu_seqlens from a CPU vision grid.

grid_thw=(T, 3) int64 CPU tensor with columns [repeat, H, W]. Output is an
int32 CPU tensor of length sum(repeat)+1 with a leading zero; each row repeats
an H*W tile ``repeat`` times. The grid is read through NumPy and not written.
"""
import statistics
import time

import pytest
import torch
import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# rows of [repeat, H, W]
COMBOS = [
    ((2, 2, 3), (1, 2, 2)),
    ((1, 1, 1), (1, 0, 5), (1, 2, 4)),
    (),
    ((3, 4, 4),),
]


def input_build(config):
    rows = tuple(tuple(row) for row in config)
    grid = torch.tensor(rows, dtype=torch.int64).reshape(-1, 3)
    return {"grid_thw": grid}


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


def compute_ref(grid_thw):
    return kernelgenbench.baseline.v1_cumulative_sequence_boundaries_grid(grid_thw)


def compute_res(grid_thw):
    return kernelgenbench.solution.v1_cumulative_sequence_boundaries_grid(grid_thw)


def expected_boundaries(config):
    values = []
    for repeat, height, width in config:
        values.extend([height * width] * repeat)
    boundaries = [0]
    total = 0
    for value in values:
        total += value
        boundaries.append(total)
    return boundaries


def compare(ref_out, res_out, config):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == torch.int32
    assert res_out.device == ref_out.device
    assert res_out.device.type == "cpu"
    assert torch.equal(ref_out, res_out)
    assert res_out.tolist() == expected_boundaries(config)
    return expected_boundaries(config)


def compute_ref_performance(**inputs):
    # CPU metadata path: wall-clock over 10 calls, reported per call in ms.
    samples = []
    for _ in range(6):
        arguments = fresh(inputs, "cpu")
        start = time.perf_counter()
        for _ in range(10):
            compute_ref(**arguments)
        samples.append((time.perf_counter() - start) * 100)
    return statistics.median(samples[1:])


def compute_res_performance(**inputs):
    samples = []
    for _ in range(6):
        arguments = fresh(inputs, "cpu")
        start = time.perf_counter()
        for _ in range(10):
            compute_res(**arguments)
        samples.append((time.perf_counter() - start) * 100)
    return statistics.median(samples[1:])


@label("v1_cumulative_sequence_boundaries_grid")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_cumulative_sequence_boundaries_grid(config):
    inputs = input_build(config)
    assert inputs["grid_thw"].device.type == "cpu"
    assert inputs["grid_thw"].dtype == torch.int64
    assert inputs["grid_thw"].shape == (len(config), 3)

    ref_inputs = fresh(inputs, "cpu")
    res_inputs = fresh(inputs, "cpu")
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config)
    for name in inputs:
        assert torch.equal(res_inputs[name], inputs[name]), f"solution changed {name}"
        assert torch.equal(ref_inputs[name], inputs[name]), f"reference changed {name}"

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )


def test_v1_cumulative_sequence_boundaries_grid_rejects_device_grid():
    if kernelgenbench.device.type != "cuda":
        pytest.skip("needs a CUDA device to build an off-CPU grid")
    grid = torch.tensor([[1, 2, 3]], dtype=torch.int64).to(kernelgenbench.device)
    for entry in (compute_ref, compute_res):
        with pytest.raises(AssertionError):
            entry(grid)
