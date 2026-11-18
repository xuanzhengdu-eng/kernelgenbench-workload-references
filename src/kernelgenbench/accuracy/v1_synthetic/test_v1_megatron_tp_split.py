"""Accuracy and performance for v1_megatron_tp_split (Megatron first-axis split).

Each case checks this rank's contiguous first-axis slice, the one-rank identity
bypass (the official helper returns the input object), the divisibility error,
and that the input is never modified.
"""
import functools

import pytest
import torch
import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.bias_activation_test_utils import fresh, measure
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (shape, dtype, world_size, rank, layout); the first axis must divide by world_size.
COMBOS = [
    (shape, dtype, world, rank, layout)
    for shape in [(4, 6), (6, 4), (8, 8), (6, 3)]
    for dtype in (torch.float32, torch.bfloat16, torch.int64)
    for world, rank in ((1, 0), (2, 0), (2, 1), (3, 0), (3, 2))
    for layout in ("contiguous", "strided")
    if shape[0] % world == 0 and (layout == "contiguous" or shape[1] > 1)
]


def input_build(config):
    shape, dtype, world, rank, layout = config
    values = torch.arange(shape[0] * shape[1], dtype=torch.int64).reshape(shape).to(dtype)
    if layout == "strided":
        # Same shape, column-major strides: narrow(0).contiguous() must copy.
        values = values.transpose(0, 1).contiguous().transpose(0, 1)
    return {"input_": values}


def compute_ref(input_, world_size, rank):
    return kernelgenbench.baseline.v1_megatron_tp_split(input_, world_size, rank)


def compute_res(input_, world_size, rank):
    return kernelgenbench.solution.v1_megatron_tp_split(input_, world_size, rank)


def expected_slice(source, world_size, rank):
    size = source.shape[0] // world_size
    return source.narrow(0, rank * size, size).contiguous()


def compare(ref_out, res_out, ref_source, res_source, config, dtype):
    shape, world, rank = config[0], config[2], config[3]
    expected = expected_slice(ref_source, world, rank)
    for name, output, source in (("reference", ref_out, ref_source), ("solution", res_out, res_source)):
        assert isinstance(output, torch.Tensor), f"{name} must return a tensor"
        assert output.shape == (shape[0] // world, shape[1]), f"{name} wrong slice shape"
        assert output.dtype == dtype, f"{name} dtype {output.dtype} != {dtype}"
        assert torch.equal(output, expected.to(output.device)), f"{name} values differ from the official slice"
        if world == 1:
            # Official bypass: identity, not a copy.
            assert output is source, f"{name} must return the input object when world_size == 1"
        else:
            assert output.is_contiguous(), f"{name} output must be contiguous"
    assert torch._C._is_alias_of(ref_out, ref_source) == torch._C._is_alias_of(res_out, res_source), \
        "reference and solution disagree on input aliasing"


def compute_ref_performance(input_, world_size, rank):
    return measure(
        functools.partial(compute_ref, world_size=world_size, rank=rank), {"input_": input_}
    )


def compute_res_performance(input_, world_size, rank):
    return measure(
        functools.partial(compute_res, world_size=world_size, rank=rank), {"input_": input_}
    )


@label("v1_megatron_tp_split")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_tp_split(config):
    shape, dtype, world, rank, layout = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    for side in (ref_inputs, res_inputs):
        assert side["input_"].shape == shape
        assert side["input_"].is_contiguous() == (layout == "contiguous")

    ref_out = compute_ref(**ref_inputs, world_size=world, rank=rank)
    res_out = compute_res(**res_inputs, world_size=world, rank=rank)
    compare(ref_out, res_out, ref_inputs["input_"], res_inputs["input_"], config, dtype)
    for side in (ref_inputs, res_inputs):
        torch.testing.assert_close(side["input_"].cpu(), inputs["input_"], rtol=0, atol=0)

    if world > 1:
        # A first dimension that does not divide is a documented rejection
        # (the official helper asserts); both sides must raise the same way.
        bad_rows = shape[0] + 1
        assert bad_rows % world != 0
        bad = torch.zeros((bad_rows, shape[1]), dtype=dtype, device=kernelgenbench.device)
        for name, entry in (("reference", compute_ref), ("solution", compute_res)):
            with pytest.raises(AssertionError):
                entry(bad, world_size=world, rank=rank)

    ref_time = compute_ref_performance(**inputs, world_size=world, rank=rank)
    res_time = compute_res_performance(**inputs, world_size=world, rank=rank)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
