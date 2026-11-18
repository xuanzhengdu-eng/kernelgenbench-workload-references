"""Accuracy and timing for DeepSpeed Random-LTD scatter.

all_activations=(B,S,C) with batch_first=True or (S,B,C) otherwise;
layer_activations=(B,R,C) / (R,B,C); sorted_indices=(B,R) int32.
The result is a fresh buffer: a copy of all_activations with the retained
positions overwritten. Inputs are read-only.
"""
import statistics
import time

import pytest
import torch
import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (batch, sequence, retained, channels, dtype, batch_first)
COMBOS = [
    (2, 8, 3, 16, torch.float32, True),
    (2, 8, 3, 16, torch.float16, True),
    (3, 16, 5, 32, torch.float32, False),
    (3, 16, 5, 32, torch.float16, False),
    (2, 12, 4, 64, torch.float16, True),
    (2, 12, 4, 64, torch.float32, False),
]


def input_build(config):
    batch, sequence, retained, channels, dtype, batch_first = config
    generator = torch.Generator(device="cpu").manual_seed(20261111)
    full_shape = (batch, sequence, channels) if batch_first else (sequence, batch, channels)
    layer_shape = (batch, retained, channels) if batch_first else (retained, batch, channels)
    indices = torch.tensor(
        [sorted((j * 2 + i) % sequence for j in range(retained)) for i in range(batch)],
        dtype=torch.int32,
    )
    return {
        "all_activations": torch.randn(full_shape, dtype=dtype, generator=generator),
        "layer_activations": torch.randn(layer_shape, dtype=dtype, generator=generator),
        "sorted_indices": indices,
        "batch_first": batch_first,
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


def compute_ref(all_activations, layer_activations, sorted_indices, batch_first):
    return kernelgenbench.baseline.v1_random_ltd_scatter(
        all_activations, layer_activations, sorted_indices, batch_first
    )


def compute_res(all_activations, layer_activations, sorted_indices, batch_first):
    return kernelgenbench.solution.v1_random_ltd_scatter(
        all_activations, layer_activations, sorted_indices, batch_first
    )


def expected_scatter(all_activations, layer_activations, sorted_indices, batch_first):
    result = all_activations.clone()
    rows = result if batch_first else result.transpose(0, 1)
    updates = layer_activations if batch_first else layer_activations.transpose(0, 1)
    index = sorted_indices.to(torch.int64).unsqueeze(-1).expand_as(updates)
    rows.scatter_(1, index, updates)
    return result


def compare(ref_out, res_out, config, ref_inputs, res_inputs):
    batch, sequence, retained, channels, dtype, batch_first = config
    full_shape = (batch, sequence, channels) if batch_first else (sequence, batch, channels)
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape == full_shape
    assert res_out.dtype == ref_out.dtype == dtype
    assert torch.equal(res_out, ref_out)
    # A clone is scattered, so the result owns storage of its own.
    assert res_out.untyped_storage().data_ptr() != res_inputs["all_activations"].untyped_storage().data_ptr()
    expected = expected_scatter(
        res_inputs["all_activations"],
        res_inputs["layer_activations"],
        res_inputs["sorted_indices"],
        batch_first,
    )
    assert torch.equal(res_out, expected)
    for name in ("all_activations", "layer_activations", "sorted_indices"):
        assert torch.equal(ref_inputs[name].cpu(), res_inputs[name].cpu())
    return expected


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_random_ltd_scatter")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_random_ltd_scatter(config):
    batch, sequence, retained, channels, dtype, batch_first = config
    inputs = input_build(config)
    full_shape = (batch, sequence, channels) if batch_first else (sequence, batch, channels)
    layer_shape = (batch, retained, channels) if batch_first else (retained, batch, channels)
    assert inputs["all_activations"].shape == full_shape
    assert inputs["layer_activations"].shape == layer_shape

    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config, ref_inputs, res_inputs)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )
