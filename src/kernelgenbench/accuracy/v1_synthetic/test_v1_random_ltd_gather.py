"""Accuracy, aliasing and timing for DeepSpeed Random-LTD gather.

activations=(B,S,C) with batch_first=True or (S,B,C) otherwise;
sorted_indices=(B,R) int32, unique and sorted; C dense and 16-byte aligned.
Output=(activations, gathered). The first element is the caller's tensor and
the second a fresh (B,R,C) / (R,B,C) tensor; neither input is written.
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
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    shape = (batch, sequence, channels) if batch_first else (sequence, batch, channels)
    indices = torch.tensor(
        [sorted((j * 2 + i) % sequence for j in range(retained)) for i in range(batch)],
        dtype=torch.int32,
    )
    return {
        "activations": torch.randn(shape, dtype=dtype, generator=generator),
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


def compute_ref(activations, sorted_indices, batch_first):
    return kernelgenbench.baseline.v1_random_ltd_gather(
        activations, sorted_indices, batch_first
    )


def compute_res(activations, sorted_indices, batch_first):
    return kernelgenbench.solution.v1_random_ltd_gather(
        activations, sorted_indices, batch_first
    )


def expected_gather(activations, sorted_indices, batch_first):
    rows = activations if batch_first else activations.transpose(0, 1)
    index = sorted_indices.to(torch.int64)
    picked = [rows[i].index_select(0, index[i]) for i in range(rows.shape[0])]
    stacked = torch.stack(picked, dim=0)
    return stacked if batch_first else stacked.transpose(0, 1).contiguous()


def compare(ref_out, res_out, config, ref_inputs, res_inputs):
    batch, sequence, retained, channels, dtype, batch_first = config
    assert isinstance(res_out, tuple) and len(res_out) == 2
    assert isinstance(ref_out, tuple) and len(ref_out) == 2
    # The forward hands back the caller's activation storage, not a copy. An
    # autograd forward returns a differentiable view of it, so the contract is
    # storage alias rather than Python object identity.
    assert torch._C._is_alias_of(res_out[0], res_inputs["activations"])
    assert torch._C._is_alias_of(ref_out[0], ref_inputs["activations"])
    assert res_out[0].untyped_storage().data_ptr() == res_inputs["activations"].untyped_storage().data_ptr()
    gathered = res_out[1]
    shape = (batch, retained, channels) if batch_first else (retained, batch, channels)
    assert gathered.shape == ref_out[1].shape == shape
    assert gathered.dtype == ref_out[1].dtype == dtype
    assert torch.equal(gathered, ref_out[1])
    assert not torch._C._is_alias_of(gathered, res_inputs["activations"])
    expected = expected_gather(
        res_inputs["activations"], res_inputs["sorted_indices"], batch_first
    )
    assert torch.equal(gathered, expected)
    for name in ("activations", "sorted_indices"):
        assert torch.equal(ref_inputs[name].cpu(), res_inputs[name].cpu())
    return gathered


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_random_ltd_gather")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_random_ltd_gather(config):
    batch, sequence, retained, channels, dtype, batch_first = config
    inputs = input_build(config)
    rows = batch if batch_first else sequence
    assert inputs["activations"].shape == (
        (batch, sequence, channels) if batch_first else (sequence, batch, channels)
    )
    assert inputs["sorted_indices"].shape == (batch, retained)
    assert inputs["sorted_indices"].dtype == torch.int32

    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config, ref_inputs, res_inputs)
    # The retained indices address the sequence axis, not the batch axis.
    assert res_out[1].shape[1 if batch_first else 0] == retained
    assert res_out[1].shape[1 if batch_first else 0] != rows

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )
