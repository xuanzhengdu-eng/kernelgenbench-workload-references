"""Accuracy, padding state and timing for v1_ds_pad_qkv.

Inputs: query=(B,H,Q,D), key=value=(B,H,K,D), FP16 or BF16, D divisible by 8,
Q divisible by 16, K<=128. Output: three tensors. Query keeps Q positions and
key/value are padded to 128 positions; every head is padded to 32/64/128.
Padding entries are exactly zero and inputs are read-only.
"""
import statistics
import time

import pytest
import torch
import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (batch, heads, query sequence, KV sequence, head width, dtype)
COMBOS = [
    (1, 2, 16, 11, 24, torch.float16),
    (1, 2, 32, 16, 32, torch.bfloat16),
    (1, 1, 16, 7, 64, torch.float16),
    (2, 3, 16, 12, 16, torch.bfloat16),
    (1, 2, 32, 128, 128, torch.float16),
]


def input_build(config):
    batch, heads, query_seq, kv_seq, head_size, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(6123)
    shape_q = (batch, heads, query_seq, head_size)
    shape_kv = (batch, heads, kv_seq, head_size)
    return {
        "query": torch.randn(shape_q, dtype=dtype, generator=generator),
        "key": torch.randn(shape_kv, dtype=dtype, generator=generator),
        "value": torch.randn(shape_kv, dtype=dtype, generator=generator),
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


def compute_ref(query, key, value):
    return kernelgenbench.baseline.v1_ds_pad_qkv(query, key, value)


def compute_res(query, key, value):
    return kernelgenbench.solution.v1_ds_pad_qkv(query, key, value)


def padded_head_size(head_size):
    return 32 if head_size < 32 else 64 if head_size < 64 else 128


def compare(ref_out, res_out, config):
    batch, heads, query_seq, kv_seq, head_size, dtype = config
    padded = padded_head_size(head_size)
    assert isinstance(res_out, list) and len(res_out) == 3
    assert isinstance(ref_out, list) and len(ref_out) == 3
    expected_shapes = [
        (batch, heads, query_seq, padded),
        (batch, heads, 128, padded),
        (batch, heads, 128, padded),
    ]
    for index, (actual, expected) in enumerate(zip(res_out, ref_out)):
        assert actual.shape == expected.shape == expected_shapes[index]
        assert actual.dtype == expected.dtype == dtype
        assert actual.device == expected.device
        assert torch.equal(actual, expected)
    # Padding rows/columns, not only the copied prefix, must be zero.
    assert torch.count_nonzero(res_out[0][:, :, :, head_size:]) == 0
    assert torch.count_nonzero(res_out[1][:, :, kv_seq:, :]) == 0
    assert torch.count_nonzero(res_out[2][:, :, kv_seq:, :]) == 0
    # The retained prefix must be the corresponding input slice.
    assert torch.equal(res_out[0][:, :, :, :head_size], ref_out[0][:, :, :, :head_size])


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_ds_pad_qkv")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_ds_pad_qkv(config):
    batch, heads, query_seq, kv_seq, head_size, dtype = config
    inputs = input_build(config)
    assert inputs["query"].shape == (batch, heads, query_seq, head_size)
    assert inputs["key"].shape == inputs["value"].shape == (batch, heads, kv_seq, head_size)
    assert inputs["query"].is_contiguous() and inputs["key"].is_contiguous()

    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config)

    # The retained prefix equals the inputs, so check it independently of the
    # reference as well; padding is the only added content.
    padded = padded_head_size(head_size)
    assert torch.equal(
        res_out[0][:, :, :, :head_size], inputs["query"].to(res_out[0].device)
    )
    assert torch.equal(
        res_out[1][:, :, :kv_seq, :head_size], inputs["key"].to(res_out[1].device)
    )
    assert torch.equal(
        res_out[2][:, :, :kv_seq, :head_size], inputs["value"].to(res_out[2].device)
    )
    for name in inputs:
        assert torch.equal(ref_inputs[name].cpu(), inputs[name]), f"reference changed {name}"
        assert torch.equal(res_inputs[name].cpu(), inputs[name]), f"solution changed {name}"

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )


def test_v1_ds_pad_qkv_rejects_unsupported_inputs():
    generator = torch.Generator(device="cpu").manual_seed(7)
    good_q = torch.randn((1, 2, 16, 24), dtype=torch.float16, generator=generator)
    good_k = torch.randn((1, 2, 8, 24), dtype=torch.float16, generator=generator)
    bad = [
        # head width not a multiple of eight
        (torch.randn((1, 2, 16, 20), dtype=torch.float16, generator=generator), good_k, good_k),
        # query sequence not a multiple of sixteen
        (torch.randn((1, 2, 12, 24), dtype=torch.float16, generator=generator), good_k, good_k),
        # KV sequence beyond 128
        (good_q, torch.randn((1, 2, 129, 24), dtype=torch.float16, generator=generator),
         torch.randn((1, 2, 129, 24), dtype=torch.float16, generator=generator)),
        # float32 has no usable pad_head_seq body at the pinned commit
        (good_q.float(), good_k.float(), good_k.float()),
    ]
    # The C++ launcher performs no validation (it trusts the workspace size), so
    # only the reference rejects these inputs; the contract records that gap.
    for query, key, value in bad:
        moved = [tensor.to(kernelgenbench.device) for tensor in (query, key, value)]
        with pytest.raises(AssertionError):
            compute_ref(*moved)
