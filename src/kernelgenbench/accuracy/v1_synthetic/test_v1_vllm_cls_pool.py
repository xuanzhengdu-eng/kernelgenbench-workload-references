"""Accuracy, error boundary and timing for vLLM's CLS-token pooling entry.

hidden_states=(sum(lengths), D); prompt_lens_cpu=(B,) int64 CPU vector.
Output=(B, D): the first row of every packed sequence, a fresh tensor.
A cursor flagged partial_prefill raises RuntimeError before any indexing, which
is the contract CLSPool has and LastPool does not.
"""
import statistics
import time

import pytest
import torch
import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (sequence lengths, hidden width, dtype): the CLS entry only accepts fully
# scheduled prompts, so every registered case keeps partial_prefill False.
COMBOS = [
    ((2, 1, 3), 5, torch.float32),
    ((3, 0, 2), 7, torch.float16),
    ((1,), 4, torch.bfloat16),
    ((2, 3), 8, torch.float32),
    ((1, 1, 1), 16, torch.float16),
]


def input_build(config):
    lengths, width, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(20261122)
    return {
        "hidden_states": torch.randn((sum(lengths), width), dtype=dtype, generator=generator),
        "prompt_lens_cpu": torch.tensor(lengths, dtype=torch.int64),
        "partial_prefill": False,
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


def compute_ref(hidden_states, prompt_lens_cpu, partial_prefill=False):
    return kernelgenbench.baseline.v1_vllm_cls_pool(
        hidden_states, prompt_lens_cpu, partial_prefill
    )


def compute_res(hidden_states, prompt_lens_cpu, partial_prefill=False):
    return kernelgenbench.solution.v1_vllm_cls_pool(
        hidden_states, prompt_lens_cpu, partial_prefill
    )


def first_indices(lengths):
    """Exclusive prefix sums, as the cursor stores first_token_indices_gpu."""
    offsets = [0]
    for length in lengths[:-1]:
        offsets.append(offsets[-1] + length)
    return offsets


def compare(ref_out, res_out, config):
    lengths, width, dtype = config
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape == (len(lengths), width)
    assert res_out.dtype == ref_out.dtype == dtype
    assert res_out.device == ref_out.device
    assert torch.equal(ref_out, res_out)
    return first_indices(lengths)


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_vllm_cls_pool")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_cls_pool(config):
    lengths, width, dtype = config
    inputs = input_build(config)
    assert inputs["hidden_states"].shape == (sum(lengths), width)
    assert inputs["prompt_lens_cpu"].device.type == "cpu"

    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    indices = compare(ref_out, res_out, config)

    packed = inputs["hidden_states"].to(res_out.device)
    torch.testing.assert_close(res_out, packed[indices], rtol=0, atol=0)
    if any(length > 1 for length in lengths):
        # The CLS index is not the LAST index, so the two entries cannot be
        # swapped without a numeric mismatch.
        last = [offset + length - 1 for offset, length in zip(indices, lengths)]
        assert indices != last
    assert not torch._C._is_alias_of(res_out, res_inputs["hidden_states"])
    for name in ("hidden_states", "prompt_lens_cpu"):
        assert torch.equal(ref_inputs[name].cpu(), inputs[name]), f"reference changed {name}"
        assert torch.equal(res_inputs[name].cpu(), inputs[name]), f"solution changed {name}"

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )


def test_v1_vllm_cls_pool_partial_prefill_rejected():
    # The CLS boundary: a partially scheduled prompt raises before any indexing,
    # and this is what distinguishes the entry from v1_vllm_last_pool.
    inputs = input_build(((2, 1), 4, torch.float16))
    hidden = inputs["hidden_states"].to(kernelgenbench.device)
    for entry in (compute_ref, compute_res):
        with pytest.raises(RuntimeError, match="partial prefill is not supported with CLS pooling"):
            entry(hidden, inputs["prompt_lens_cpu"], True)
