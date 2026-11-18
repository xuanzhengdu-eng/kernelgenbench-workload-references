"""Accuracy, state and timing for v1_vllm_mean_pool.

Inputs: packed hidden_states=(sum(lengths), D) and a CPU int64 length vector.
Output: (len(lengths), D) FP32 on the hidden-state device. A zero length makes
that row NaN; an empty batch is (0, D); partial prefill raises RuntimeError.
"""
import statistics
import time

import pytest
import torch
import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (prompt lengths, feature width, non-contiguous packed rows, dtype)
COMBOS = [
    ((2, 1, 3), 5, False, torch.float32),
    ((3, 0, 2), 7, False, torch.float16),
    ((1,), 4, True, torch.bfloat16),
    ((0, 0), 3, False, torch.float32),
    ((), 6, False, torch.float16),
]


def input_build(config):
    lengths, width, strided, dtype = config
    rows = sum(lengths)
    generator = torch.Generator(device="cpu").manual_seed(20261110)
    hidden = torch.randn((rows, width), dtype=dtype, generator=generator)
    if strided and rows:
        # Every other column: row stride 2*width, element stride 2.
        backing = torch.zeros((rows, 2 * width), dtype=dtype)
        backing[:, ::2] = hidden
        hidden = backing[:, ::2]
    return {
        "hidden_states": hidden,
        "prompt_lens_cpu": torch.tensor(lengths, dtype=torch.int64),
        "partial_prefill": False,
    }


def fresh(inputs, device):
    return {
        name: value.to(device=device, copy=True) if isinstance(value, torch.Tensor) else value
        for name, value in inputs.items()
    }


def measure(fn, inputs):
    """Median single-call milliseconds; transfer and warmup are outside the timer."""
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


def compute_ref(hidden_states, prompt_lens_cpu, partial_prefill=False):
    return kernelgenbench.baseline.v1_vllm_mean_pool(
        hidden_states, prompt_lens_cpu, partial_prefill
    )


def compute_res(hidden_states, prompt_lens_cpu, partial_prefill=False):
    return kernelgenbench.solution.v1_vllm_mean_pool(
        hidden_states, prompt_lens_cpu, partial_prefill
    )


def compare(ref_out, res_out, dtype):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.dtype == torch.float32 and ref_out.dtype == torch.float32
    assert res_out.shape == ref_out.shape
    assert res_out.device == ref_out.device
    torch.testing.assert_close(ref_out, res_out, rtol=1e-6, atol=1e-6, equal_nan=True)


def compute_ref_performance(hidden_states, prompt_lens_cpu, partial_prefill=False):
    return measure(
        lambda hidden: compute_ref(hidden, prompt_lens_cpu, partial_prefill),
        {"hidden": hidden_states},
    )


def compute_res_performance(hidden_states, prompt_lens_cpu, partial_prefill=False):
    return measure(
        lambda hidden: compute_res(hidden, prompt_lens_cpu, partial_prefill),
        {"hidden": hidden_states},
    )


@label("v1_vllm_mean_pool")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_mean_pool(config):
    lengths, width, strided, dtype = config
    inputs = input_build(config)
    expected_shape = (len(lengths), width)
    assert inputs["hidden_states"].shape == (sum(lengths), width)
    assert inputs["hidden_states"].is_contiguous() != strided
    assert inputs["prompt_lens_cpu"].device.type == "cpu"

    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)
    assert ref_out.shape == expected_shape and res_out.shape == expected_shape
    if kernelgenbench.device.type == "cuda":
        assert res_out.device.type == "cuda"
    assert not torch._C._is_alias_of(res_out, res_inputs["hidden_states"])
    for name in ("hidden_states", "prompt_lens_cpu"):
        assert torch.equal(ref_inputs[name].cpu(), inputs[name]), f"reference changed {name}"
        assert torch.equal(res_inputs[name].cpu(), inputs[name]), f"solution changed {name}"

    # The reference must reproduce the FP64 mean of every packed slice, with NaN
    # exactly where a sequence length is zero.
    exact = torch.full(expected_shape, float("nan"), dtype=torch.float64)
    if sum(lengths):
        moving = inputs["hidden_states"].double()
        offset = 0
        for index, length in enumerate(lengths):
            if length:
                exact[index] = moving[offset:offset + length].mean(dim=0)
            offset += length
    torch.testing.assert_close(
        ref_out.cpu().double(), exact, rtol=1e-6, atol=1e-6, equal_nan=True
    )

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )


def test_v1_vllm_mean_pool_partial_prefill_rejected():
    inputs = input_build(((2, 1), 4, False, torch.float16))
    hidden = inputs["hidden_states"].to(kernelgenbench.device)
    for entry in (compute_ref, compute_res):
        with pytest.raises(RuntimeError, match="partial prefill"):
            entry(hidden, inputs["prompt_lens_cpu"], partial_prefill=True)
