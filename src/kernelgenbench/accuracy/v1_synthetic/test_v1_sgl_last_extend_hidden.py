"""Accuracy and performance for v1_sgl_last_extend_hidden.

Input: packed hidden rows [N, W] (optionally a padded, non-contiguous view) and
one extension length per request. Output: the last hidden row of each extended
segment; when N equals the request count the identical input object is returned
as an identity fast path.

Key assertions: exact row indices (computed here with an independent Python
loop, including the documented index -1 wrap for a leading zero length), the
identity fast path object, dtype/shape/device, fresh output on the gather path,
and an unmodified input.
"""
import statistics

import pytest
import torch

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

NAME = "v1_sgl_last_extend_hidden"

# (extension lengths, packed rows, hidden width, padded row storage)
CASES = [
    ((1, 1, 1), 3, 4, 0),
    ((2, 3, 1), 6, 7, 2),
    ((0, 3, 2), 5, 4, 0),
    ((2, 0, 3), 5, 3, 2),
    ((), 0, 4, 0),
]
COMBOS = [(case, dtype)
          for case in CASES
          for dtype in (torch.float32, torch.float16, torch.bfloat16)]


def input_build(config):
    (lengths, rows, width, pad), dtype = config
    generator = torch.Generator().manual_seed(20261110)
    base = torch.randn((rows, width + pad), dtype=dtype, generator=generator)
    hidden = base[:, :width] if pad else base
    return dict(hidden_states=hidden, extend_lens=list(lengths))


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_sgl_last_extend_hidden(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_sgl_last_extend_hidden(**inputs)


def _expected_rows(lengths, rows):
    selected = []
    start = 0
    for length in lengths:
        index = start + length - 1
        if index < 0:  # Python indexing: a leading zero length selects the last row
            index += rows
        selected.append(index)
        start += length
    return selected


def compare(ref_out, res_out, hidden, config):
    (lengths, rows, width, pad), dtype = config
    for out in (ref_out, res_out):
        assert isinstance(out, torch.Tensor)
        assert out.dtype == dtype
        assert out.device == hidden.device
    assert res_out.shape == ref_out.shape
    assert torch.equal(res_out, ref_out), "selected rows differ"
    expected = hidden.index_select(
        0, torch.tensor(_expected_rows(lengths, rows), dtype=torch.int64,
                        device=hidden.device))
    assert torch.equal(res_out, expected), "row selection does not match the contract"
    if rows == len(lengths):
        assert ref_out is hidden and res_out is hidden, "identity fast path must return the input"
    else:
        assert not torch._C._is_alias_of(res_out, hidden), "gather path must allocate"
        assert res_out.is_contiguous()
    assert res_out.shape == (len(lengths), width)


def _to_device(tensor, device):
    """Move a tensor keeping its (possibly padded) strides; .to() densifies."""
    if tensor.is_contiguous():
        return tensor.to(device)
    moved = torch.empty_strided(tuple(tensor.shape), tuple(tensor.stride()),
                                dtype=tensor.dtype, device=device)
    moved.copy_(tensor)
    return moved


def _bench(fn, inputs):
    device = kernelgenbench.device
    prepared = {"hidden_states": _to_device(inputs["hidden_states"], device),
                "extend_lens": inputs["extend_lens"]}
    for _ in range(5):
        fn(**prepared)
    torch.cuda.synchronize()
    samples = []
    for _ in range(20):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(**prepared)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def compute_ref_performance(inputs):
    return _bench(compute_ref, inputs)


def compute_res_performance(inputs):
    return _bench(compute_res, inputs)


@label("v1_sgl_last_extend_hidden")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_last_extend_hidden(config):
    inputs = input_build(config)
    hidden = _to_device(inputs["hidden_states"], kernelgenbench.device)
    before = hidden.clone()
    ref_out = compute_ref(hidden_states=hidden, extend_lens=inputs["extend_lens"])
    res_out = compute_res(hidden_states=hidden, extend_lens=inputs["extend_lens"])
    compare(ref_out, res_out, hidden, config)
    assert torch.equal(before, hidden), "hidden states were modified"

    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
