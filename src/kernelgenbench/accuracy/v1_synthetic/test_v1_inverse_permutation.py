"""Accuracy and timing for DeepSpeed's inverse permutation.

permuted_indices=(M,) int32/int64 where each of 0..num_rows-1 appears exactly
once and all other entries are padding; num_rows>=0. Output=(num_rows,) int64
holding the padded position of every original row. Input is read-only and the
output is fresh.
"""
import statistics
import time

import pytest
import torch
import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (permuted indices, number of original rows, dtype)
COMBOS = [
    ([2, -1, 0, 9, 1], 3, torch.int32),
    ([3, 1, -2, 0, 4, 2], 5, torch.int64),
    ([-1, 4], 0, torch.int32),
    ([0], 1, torch.int64),
    ([4, 0, 3, 1, 2, -1, -1], 5, torch.int32),
]


def input_build(config):
    indices, num_rows, dtype = config
    return {
        "permuted_indices": torch.tensor(indices, dtype=dtype),
        "num_rows": num_rows,
    }


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


def compute_ref(permuted_indices, num_rows):
    return kernelgenbench.baseline.v1_inverse_permutation(permuted_indices, num_rows)


def compute_res(permuted_indices, num_rows):
    return kernelgenbench.solution.v1_inverse_permutation(permuted_indices, num_rows)


def expected_inverse(indices, num_rows):
    position = {}
    for slot, value in enumerate(indices):
        if 0 <= value < num_rows:
            position[value] = slot
    return [position[row] for row in range(num_rows)]


def compare(ref_out, res_out, config):
    indices, num_rows, dtype = config
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape == (num_rows,)
    assert res_out.dtype == ref_out.dtype == torch.int64
    assert res_out.device == ref_out.device
    assert torch.equal(ref_out, res_out)
    assert res_out.tolist() == expected_inverse(indices, num_rows)
    # Padding slots must not shift valid rows; a wrong implementation that
    # scatters every entry would return different positions here.
    return expected_inverse(indices, num_rows)


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_inverse_permutation")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_inverse_permutation(config):
    indices, num_rows, dtype = config
    inputs = input_build(config)
    assert inputs["permuted_indices"].shape == (len(indices),)
    assert inputs["permuted_indices"].dtype == dtype
    assert sorted(value for value in indices if 0 <= value < num_rows) == list(range(num_rows))

    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config)
    assert not torch._C._is_alias_of(res_out, res_inputs["permuted_indices"])
    for name in ("permuted_indices",):
        assert torch.equal(res_inputs[name].cpu(), inputs[name]), f"solution changed {name}"
        assert torch.equal(ref_inputs[name].cpu(), inputs[name]), f"reference changed {name}"

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )
