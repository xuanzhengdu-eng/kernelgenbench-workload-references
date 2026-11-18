"""Accuracy and performance for v1_vllm_scatter_output_slices_.

Input: a concatenated `output` tensor, the per-item output-token counts, the
order in which items are visited, and a destination that is either a list with
one slot per item or a dict. The contract: return None and write consecutive
slices of `output` into `dest[idx]`, sharing `output`'s storage unless
`clone=True`; the unused tail of `output` is never written.

Key assertions: exact slice values and placement (offset advances in `indices`
order), dest presence/None, alias status matches the `clone` flag on both
sides, and neither `output` nor the counts are modified.
"""
import copy
import statistics

import pytest
import torch

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

NAME = "v1_vllm_scatter_output_slices_"

# (output shape, per-item token counts, visit order, clone, dest kind)
COMBOS = [
    ((7, 4), (2, 0, 3), (2, 0, 1), False, "list"),
    ((5, 3, 2), (1, 3, 3, 1), (1, 3), True, "dict"),
    ((4, 8), (0, 2), (), False, "dict"),
    ((4, 5), (0, 3, 2), (0, 2), True, "list"),
    ((6,), (1, 1, 1, 1, 1, 1), (5, 4, 3, 2, 1, 0), False, "list"),
]


def input_build(config):
    shape, lengths, indices, clone, kind = config
    total = 1
    for size in shape:
        total *= size
    output = torch.arange(total, dtype=torch.float32).reshape(shape)
    dest = [None] * len(lengths) if kind == "list" else {}
    return dict(output=output, indices=list(indices),
                per_item_out_tokens=list(lengths), dest=dest, clone=clone)


def _fresh(inputs):
    copied = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            copied[key] = value.clone()
        elif key == "dest":
            copied[key] = [None] * len(value) if isinstance(value, list) else {}
        else:
            copied[key] = copy.deepcopy(value)
    return copied


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_vllm_scatter_output_slices_(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_vllm_scatter_output_slices_(**inputs)


def _slot(dest, key):
    return dest[key] if isinstance(dest, list) else dest.get(key)


def compare(ref_out, res_out, ref_inputs, res_inputs, config):
    _, lengths, indices, clone, _ = config
    assert ref_out is None and res_out is None, "scatter output must return None"
    for key, count in enumerate(lengths):
        expected = _slot(ref_inputs["dest"], key)
        actual = _slot(res_inputs["dest"], key)
        assert (expected is None) == (actual is None), f"dest[{key}] presence differs"
        if expected is None:
            continue
        assert isinstance(actual, torch.Tensor)
        assert actual.shape == expected.shape == (count, *config[0][1:])
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected), f"dest[{key}] values differ"
        assert (torch._C._is_alias_of(actual, res_inputs["output"])
                == torch._C._is_alias_of(expected, ref_inputs["output"])), \
            f"dest[{key}] alias status differs"
        assert torch._C._is_alias_of(actual, res_inputs["output"]) == (not clone), \
            f"dest[{key}] must alias output iff clone=False"
    offset = 0
    for idx in indices:
        count = lengths[idx]
        expected = res_inputs["output"][offset:offset + count]
        assert torch.equal(_slot(res_inputs["dest"], idx), expected), \
            f"dest[{idx}] is not output[{offset}:{offset + count}]"
        offset += count
    for key, value in ref_inputs.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, res_inputs[key]), f"input {key} was modified"
    assert torch.equal(ref_inputs["output"], res_inputs["output"])
    assert ref_inputs["per_item_out_tokens"] == res_inputs["per_item_out_tokens"]


def _bench(fn, inputs):
    device = kernelgenbench.device
    prepared = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in inputs.items()}
    def call():
        arguments = dict(prepared)
        dest = prepared["dest"]
        arguments["dest"] = ([None] * len(dest) if isinstance(dest, list) else {})
        return fn(**arguments)
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(20):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def compute_ref_performance(inputs):
    return _bench(compute_ref, inputs)


def compute_res_performance(inputs):
    return _bench(compute_res, inputs)


@label("v1_vllm_scatter_output_slices_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_scatter_output_slices_(config):
    inputs = input_build(config)
    ref_inputs = _fresh(inputs)
    res_inputs = _fresh(inputs)
    device = kernelgenbench.device
    for side in (ref_inputs, res_inputs):
        side["output"] = side["output"].to(device)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, ref_inputs, res_inputs, config)

    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
