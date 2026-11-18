"""Accuracy, state and timing for DeepSpeed Random-LTD gather_back.

a_gradients=(B,S,C) with batch_first=True or (S,B,C) otherwise;
g_gradients=(B,R,C) / (R,B,C); sorted_indices=(B,R) int32.
The official backward writes the retained gradients into a_gradients itself and
returns that same tensor, so the caller's buffer is modified (rows outside the
retained index keep their original values). It is an autograd backward, so the
return value is ``(updated_gradient, None, None)``.
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
    generator = torch.Generator(device="cpu").manual_seed(20261110)
    shape = (batch, sequence, channels) if batch_first else (sequence, batch, channels)
    grad_shape = (batch, retained, channels) if batch_first else (retained, batch, channels)
    indices = torch.tensor(
        [sorted((j * 2 + i) % sequence for j in range(retained)) for i in range(batch)],
        dtype=torch.int32,
    )
    return {
        "a_gradients": torch.randn(shape, dtype=dtype, generator=generator),
        "g_gradients": torch.randn(grad_shape, dtype=dtype, generator=generator),
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


def compute_ref(a_gradients, g_gradients, sorted_indices, batch_first):
    return kernelgenbench.baseline.v1_random_ltd_gather_back(
        a_gradients, g_gradients, sorted_indices, batch_first
    )


def compute_res(a_gradients, g_gradients, sorted_indices, batch_first):
    return kernelgenbench.solution.v1_random_ltd_gather_back(
        a_gradients, g_gradients, sorted_indices, batch_first
    )


def expected_scatter(original_grad, g_gradients, sorted_indices, batch_first):
    result = original_grad.clone()
    rows = result if batch_first else result.transpose(0, 1)
    updates = g_gradients if batch_first else g_gradients.transpose(0, 1)
    index = sorted_indices.to(torch.int64).unsqueeze(-1).expand_as(updates)
    rows.scatter_(1, index, updates)
    return result


def compare(ref_out, res_out, config, ref_inputs, res_inputs, original_grad):
    batch, sequence, retained, channels, dtype, batch_first = config
    shape = (batch, sequence, channels) if batch_first else (sequence, batch, channels)
    # One gradient slot per forward input: activation grad, then two non-tensor
    # arguments (sorted_indices, batch_first) that must be None.
    assert isinstance(res_out, tuple) and len(res_out) == 3
    assert isinstance(ref_out, tuple) and len(ref_out) == 3
    assert ref_out[1] is None and ref_out[2] is None
    assert res_out[1] is None and res_out[2] is None
    res_out, ref_out = res_out[0], ref_out[0]
    assert res_out.shape == ref_out.shape == shape
    assert res_out.dtype == ref_out.dtype == dtype
    # In-place: the returned tensor is the caller's own gradient buffer.
    assert res_out is res_inputs["a_gradients"]
    assert ref_out is ref_inputs["a_gradients"]
    expected = expected_scatter(
        original_grad.to(res_out.device),
        res_inputs["g_gradients"],
        res_inputs["sorted_indices"],
        batch_first,
    )
    assert torch.equal(res_out, expected)
    assert torch.equal(ref_out, expected)
    # Rows that are not retained keep the incoming gradient values.
    if torch.equal(res_out, original_grad.to(res_out.device)):
        raise AssertionError("scatter_back did not write the retained rows")
    assert torch.equal(
        ref_inputs["g_gradients"].cpu(), res_inputs["g_gradients"].cpu()
    )
    return expected


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_random_ltd_gather_back")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_random_ltd_gather_back(config):
    batch, sequence, retained, channels, dtype, batch_first = config
    inputs = input_build(config)
    assert inputs["g_gradients"].shape == (
        (batch, retained, channels) if batch_first else (retained, batch, channels)
    )
    assert inputs["a_gradients"].is_contiguous()

    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config, ref_inputs, res_inputs, inputs["a_gradients"])

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )
