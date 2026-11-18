"""Accuracy and performance for v1_sgl_sp_entry_scatter.

Item: row 313 (OP-f9d3edd93be9).  SGLang layer-norm sequence-parallel entry:
pad the token rows to a multiple of TP, take this rank's contiguous slice, and
expose the real (unpadded) token count as the module state.  ``tp_size == 1`` is
an identity that returns the input object.  Aliasing is part of the contract:
with no padding the ``.contiguous()`` slice is still a view of the input, with
padding it is a view of the padded copy.
"""
import time

import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from sandbox.utils.accuracy_utils import CustomBenchmarkResult

# (token rows, hidden width, TP size, rank, strided input)
SHAPES = [
    (0, 8, 1, 0, False),
    (0, 8, 4, 3, False),
    (1, 8, 4, 3, False),
    (7, 16, 1, 0, True),
    (7, 16, 2, 0, False),
    (7, 16, 2, 1, False),
    (8, 8, 2, 1, False),
    (8, 8, 4, 2, True),
    (9, 5, 3, 1, False),
]
COMBOS = [(spec, dtype) for spec in SHAPES
          for dtype in (torch.float32, torch.float16, torch.bfloat16)]


def input_build(config):
    spec, dtype = config
    rows, width, tp_size, rank, strided = spec
    generator = torch.Generator(device="cpu").manual_seed(20261116)
    if strided:
        base = torch.randn((rows, width * 2), dtype=dtype, generator=generator)
        tensor = base[:, ::2]
        assert tensor.shape == (rows, width) and not tensor.is_contiguous()
    else:
        tensor = torch.randn((rows, width), dtype=dtype, generator=generator)
    return {"hidden_states": tensor, "tp_size": tp_size, "tp_rank": rank}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_sgl_sp_entry_scatter(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_sgl_sp_entry_scatter(**inputs)


def expected_slice(tensor, tp_size, rank):
    """Independent oracle: zero-pad to a multiple of TP, then take the chunk."""
    rows, width = tensor.shape
    padded = -(-rows // tp_size) * tp_size
    if padded > rows:
        tensor = torch.cat(
            (tensor, torch.zeros((padded - rows, width), dtype=tensor.dtype,
                                 device=tensor.device)), dim=0)
    per = padded // tp_size
    return tensor[rank * per:(rank + 1) * per].contiguous()


def compare(ref_out, res_out, dtype):
    assert isinstance(res_out, tuple) and len(res_out) == 2, type(res_out)
    assert isinstance(res_out[0], torch.Tensor)
    ref_tensor, ref_count = ref_out
    res_tensor, res_count = res_out
    assert res_count == ref_count and isinstance(res_count, int)
    assert res_tensor.shape == ref_tensor.shape
    assert res_tensor.dtype == ref_tensor.dtype == dtype
    torch.testing.assert_close(res_tensor, ref_tensor, rtol=0, atol=0)


@label("v1_sgl_sp_entry_scatter")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_sp_entry_scatter(config):
    spec, dtype = config
    rows, width, tp_size, rank, _ = spec
    device = kernelgenbench.device
    inputs = {name: value.to(device) if isinstance(value, torch.Tensor) else value
              for name, value in input_build(config).items()}
    ref_inputs = {name: value.clone() if isinstance(value, torch.Tensor) else value
                  for name, value in inputs.items()}
    res_inputs = {name: value.clone() if isinstance(value, torch.Tensor) else value
                  for name, value in inputs.items()}

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    ref_tensor, ref_count = ref_out
    res_tensor, res_count = res_out
    assert res_count == rows, (res_count, rows)
    per = (-(-rows // tp_size))
    assert tuple(res_tensor.shape) == (per, width), res_tensor.shape
    # Values: this rank's slice of the zero-padded rows.
    oracle = expected_slice(inputs["hidden_states"], tp_size, rank)
    torch.testing.assert_close(res_tensor, oracle, rtol=0, atol=0)
    if tp_size > 1 and rows % tp_size:
        # The padding rows are zeros and are only present in the last slice.
        pad = per * tp_size - rows
        if rank == tp_size - 1:
            assert pad > 0
            torch.testing.assert_close(
                res_tensor[-pad:], torch.zeros_like(res_tensor[-pad:]))
    # Identity branch keeps the exact input object; padded branch cannot alias it.
    aliases_source = torch._C._is_alias_of(res_tensor, res_inputs["hidden_states"])
    ref_aliases = torch._C._is_alias_of(ref_tensor, ref_inputs["hidden_states"])
    assert aliases_source == ref_aliases, (aliases_source, ref_aliases)
    if tp_size == 1:
        assert res_tensor is res_inputs["hidden_states"]
        assert ref_tensor is ref_inputs["hidden_states"]
    if tp_size > 1 and rows % tp_size:
        assert not aliases_source, "padded output must not alias the input"
    # Neither side modifies the input.
    for side in (ref_inputs, res_inputs):
        torch.testing.assert_close(side["hidden_states"], inputs["hidden_states"],
                                   rtol=0, atol=0)

    def measure(fn, kwargs, iters=50):
        for _ in range(5):
            fn(**kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            for _ in range(iters):
                fn(**kwargs)
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / iters
        begin = time.perf_counter()
        for _ in range(iters):
            fn(**kwargs)
        return (time.perf_counter() - begin) * 1000 / iters

    ref_time = measure(compute_ref, ref_inputs)
    res_time = measure(compute_res, res_inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
