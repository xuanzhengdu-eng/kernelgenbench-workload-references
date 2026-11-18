"""Accuracy and timing for v1_vision_patch_attention_mask on the pinned SGLang entry.

The candidate binds ``VisionSdpaAttention.generate_patch_attention_mask``.  It
returns None without cu_seqlens, a [1, s, s] block-diagonal boolean mask when
flatten_batch is set, and a padded-batch [batch, 1, s, s] mask otherwise.  The
upstream mask factory memoizes on (s, flatten_batch, cu_seqlens tuple), so the
test asks both sides for the same mask twice and compares values, not storage.
"""
import statistics

import pytest
import torch

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (maximum/padded sequence length, cumulative boundaries or None, flatten)
CASES = [
    (8, (0, 3, 8), True),
    (6, (0, 2, 2, 5), False),
    (5, (0, 5), False),
    (4, None, True),
    (0, (0, 0), True),
    (8, (0, 4, 8), False),
]
COMBOS = [(index, case, dtype) for index, case in enumerate(CASES)
          for dtype in (torch.int32, torch.int64)]


def fresh(inputs, device):
    copied = {}
    for name, value in inputs.items():
        if not isinstance(value, torch.Tensor):
            copied[name] = value
            continue
        target = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype,
                                     device=device)
        target.copy_(value)
        copied[name] = target
    return copied


def measure(fn, inputs, device):
    for _ in range(3):
        fn(**fresh(inputs, device))
    torch.cuda.synchronize()
    samples = []
    for _ in range(6):
        arguments = fresh(inputs, device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(**arguments)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples[1:])


def input_build(config):
    _, case, dtype = config
    length, boundaries, flatten = case
    return {
        "s": length,
        "cu_seqlens": (torch.tensor(boundaries, dtype=dtype)
                       if boundaries is not None else None),
        "flatten_batch": flatten,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_vision_patch_attention_mask(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_vision_patch_attention_mask(**inputs)


def compare(ref_out, res_out, case, data):
    length, boundaries, flatten = case
    if boundaries is None:
        assert ref_out is None and res_out is None
        return
    assert isinstance(res_out, torch.Tensor) and isinstance(ref_out, torch.Tensor)
    sequences = len(boundaries) - 1
    expected_shape = (1, length, length) if flatten else (sequences, 1, length, length)
    assert res_out.shape == ref_out.shape == expected_shape
    assert res_out.dtype == ref_out.dtype == torch.bool
    assert res_out.device == ref_out.device
    assert torch.equal(res_out, ref_out)
    # Independent expectation from the documented block/padding rule.
    expected = torch.zeros(expected_shape, dtype=torch.bool)
    if flatten:
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            expected[..., start:end, start:end] = True
    else:
        lengths = [end - start for start, end in zip(boundaries[:-1], boundaries[1:])]
        for index, length_i in enumerate(lengths):
            expected[index, 0, :length_i, :length_i] = True
    assert torch.equal(res_out, expected), "independent mask rule differs"


def compute_ref_performance(inputs):
    return measure(compute_ref, inputs, kernelgenbench.device)


def compute_res_performance(inputs):
    return measure(compute_res, inputs, kernelgenbench.device)


@label("v1_vision_patch_attention_mask")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vision_patch_attention_mask(config):
    case = config[1]
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, case, inputs)

    # Repeated calls on the (memoizing) upstream entry keep the same values.
    again = compute_res(**res_inputs)
    if res_out is None:
        assert again is None
    else:
        assert torch.equal(again, res_out)
    if ref_inputs["cu_seqlens"] is not None:
        assert torch.equal(ref_inputs["cu_seqlens"].cpu(),
                           res_inputs["cu_seqlens"].cpu())

    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
