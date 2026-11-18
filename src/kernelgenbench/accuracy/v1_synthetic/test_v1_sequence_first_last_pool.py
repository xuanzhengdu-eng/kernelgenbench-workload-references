"""Accuracy and timing for SGLang's single pooling entry (real PoolingType enum).

hidden_states=(sum(lengths), D); extend_seq_lens=(B,) int32 device vector;
pooling_type is the genuine ``sglang.srt.layers.pooler.PoolingType`` member, so
the same three enum values the upstream method dispatches on reach both sides.
Output=(B, D), hidden-state dtype and device, fresh storage.
"""
import statistics
import time

import pytest
import torch
import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (sequence lengths, hidden width, PoolingType member name, dtype)
# MEAN only gets positive lengths here: the upstream cumsum-difference formula
# divides by the length, so a zero length is the separate NaN case already
# covered by v1_vllm_mean_pool's entry (different formula, different item).
COMBOS = [
    ((2, 3, 1), 8, "LAST", torch.float32),
    ((4, 4), 32, "LAST", torch.bfloat16),
    ((2, 3, 1), 8, "CLS", torch.float32),
    ((1, 1, 1), 8, "CLS", torch.float16),
    ((2, 3, 1), 16, "MEAN", torch.float32),
    ((4, 4), 32, "MEAN", torch.float16),
]


def pooling_type(name):
    """Resolve the real enum member at call time (module import stays framework-free)."""
    from sglang.srt.layers.pooler import PoolingType

    return PoolingType[name]


def input_build(config):
    lengths, width, name, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    return {
        "hidden_states": torch.randn((sum(lengths), width), dtype=dtype, generator=generator),
        "extend_seq_lens": torch.tensor(lengths, dtype=torch.int32),
        "pooling_type": pooling_type(name),
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


def compute_ref(hidden_states, extend_seq_lens, pooling_type):
    return kernelgenbench.baseline.v1_sequence_first_last_pool(
        hidden_states, extend_seq_lens, pooling_type
    )


def compute_res(hidden_states, extend_seq_lens, pooling_type):
    return kernelgenbench.solution.v1_sequence_first_last_pool(
        hidden_states, extend_seq_lens, pooling_type
    )


def offsets_of(lengths):
    offsets = [0]
    for length in lengths[:-1]:
        offsets.append(offsets[-1] + length)
    return offsets


def expected_rows(lengths, name):
    offsets = offsets_of(lengths)
    if name == "LAST":
        return [offset + length - 1 for offset, length in zip(offsets, lengths)]
    if name == "CLS":
        return offsets
    return None


def compare(ref_out, res_out, config, packed):
    lengths, width, name, dtype = config
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape == (len(lengths), width)
    assert res_out.dtype == ref_out.dtype == dtype
    assert res_out.device == ref_out.device
    assert torch.equal(ref_out, res_out)

    if name in ("LAST", "CLS"):
        rows = expected_rows(lengths, name)
        torch.testing.assert_close(res_out, packed[rows], rtol=0, atol=0)
        return rows
    # MEAN: compare against an FP64 segment mean of the packed slices.
    exact = []
    offset = 0
    for length in lengths:
        exact.append(packed[offset:offset + length].double().mean(dim=0))
        offset += length
    exact = torch.stack(exact)
    rtol, atol = (2e-3, 1e-3) if dtype == torch.float16 else (4e-6, 2e-6)
    torch.testing.assert_close(res_out.cpu().double(), exact.cpu(), rtol=rtol, atol=atol)
    return None


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_sequence_first_last_pool")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sequence_first_last_pool(config):
    lengths, width, name, dtype = config
    inputs = input_build(config)
    assert inputs["hidden_states"].shape == (sum(lengths), width)
    assert inputs["extend_seq_lens"].dtype == torch.int32
    # The parameter really is the upstream enum, not a local string selector.
    assert type(inputs["pooling_type"]).__name__ == "PoolingType"
    assert int(inputs["pooling_type"]) == {"LAST": 0, "CLS": 1, "MEAN": 2}[name]

    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config, inputs["hidden_states"].to(res_out.device))

    assert not torch._C._is_alias_of(res_out, res_inputs["hidden_states"])
    if name in ("LAST", "CLS"):
        # LAST and CLS pick different rows whenever a sequence is longer than one
        # token; MEAN is the only mode whose result differs from both.
        if any(length > 1 for length in lengths):
            assert expected_rows(lengths, "LAST") != expected_rows(lengths, "CLS")
    for tensor_name in ("hidden_states", "extend_seq_lens"):
        assert torch.equal(ref_inputs[tensor_name].cpu(), inputs[tensor_name]), (
            f"reference changed {tensor_name}"
        )
        assert torch.equal(res_inputs[tensor_name].cpu(), inputs[tensor_name]), (
            f"solution changed {tensor_name}"
        )

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )


def test_v1_sequence_first_last_pool_rejects_unknown_type():
    # The enum has only three members, so the unsupported-type path is reached
    # with a name the upstream dispatch does not implement.
    class _Fake:
        name = "MEDIAN"

    inputs = input_build(((2, 1), 4, "LAST", torch.float16))
    hidden = inputs["hidden_states"].to(kernelgenbench.device)
    lens = inputs["extend_seq_lens"].to(kernelgenbench.device)
    with pytest.raises(ValueError, match="Unsupported pooling type"):
        compute_ref(hidden, lens, _Fake())
