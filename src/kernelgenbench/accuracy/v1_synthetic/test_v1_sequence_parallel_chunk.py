"""Accuracy and performance for v1_sequence_parallel_chunk (vLLM, pinned f358f057).

The reference is the scalar-input sub-contract of the official
``sequence_parallel_chunk_impl``: the TP process group is replaced by explicit
``tp_size``/``tp_rank`` scalars.  Inputs are built on CPU and copied to
``kernelgenbench.device``; the official implementation is called on the same
device through the Solution binding.

Per case the contract is:
  * rows are zero-padded up to a multiple of ``tp_size``,
  * this rank gets a contiguous ``padded_rows / tp_size``-row chunk,
  * the output values are exact (pure copy/pad, no arithmetic),
  * padding rows inside the selected chunk are all zero,
  * the output never aliases the input and the input is never modified.
"""
import functools
import math

import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.bias_activation_test_utils import fresh, measure
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# Token-grid shape (B, T, H); the 2-D input handed to the operator is (B*T, H).
# (2, 16, 128) and (2, 128, 1024) are already TP-2 aligned; (3, 11, 128) is the
# odd row count that forces the zero-padded branch for tp_rank 1.
COMBOS = [
    (shape, dtype, 2, tp_rank)
    for shape in [(2, 16, 128), (3, 11, 128), (2, 128, 1024)]
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
    for tp_rank in (0, 1)
]


def input_build(config):
    shape, dtype, tp_size, tp_rank = config
    rows, hidden = math.prod(shape[:-1]), shape[-1]
    generator = torch.Generator(device="cpu").manual_seed(20261115)
    x = torch.randn((rows, hidden), dtype=dtype, generator=generator, device="cpu")
    return {"x": x}


def compute_ref(x, tp_size, tp_rank):
    return kernelgenbench.baseline.v1_sequence_parallel_chunk(
        x, tp_size=tp_size, tp_rank=tp_rank
    )


def compute_res(x, tp_size, tp_rank):
    return kernelgenbench.solution.v1_sequence_parallel_chunk(
        x, tp_size=tp_size, tp_rank=tp_rank
    )


def expected_chunk(x, tp_size, tp_rank):
    """Independent construction of the padded, rank-local chunk."""
    rows, hidden = x.shape
    padded_rows = ((rows + tp_size - 1) // tp_size) * tp_size
    chunk = padded_rows // tp_size
    start = tp_rank * chunk
    out = x.new_zeros((chunk, hidden))
    available = min(chunk, max(rows - start, 0))
    if available:
        out[:available] = x[start:start + available]
    return out


def compare(ref_out, res_out, ref_source, res_source, config, dtype):
    shape, tp_size, tp_rank = config[0], config[2], config[3]
    rows, hidden = math.prod(shape[:-1]), shape[-1]
    padded_rows = ((rows + tp_size - 1) // tp_size) * tp_size
    chunk = padded_rows // tp_size
    start = tp_rank * chunk
    expected = expected_chunk(ref_source.detach(), tp_size, tp_rank)

    for name, output, source in (
        ("reference", ref_out, ref_source),
        ("solution", res_out, res_source),
    ):
        assert isinstance(output, torch.Tensor), f"{name} must return a tensor"
        assert output.shape == (chunk, hidden), f"{name} shape {tuple(output.shape)}"
        assert output.dtype == dtype, f"{name} dtype {output.dtype} != {dtype}"
        assert output.device == source.device, f"{name} moved device"
        # Pure copy + zero padding: bit-exact, no tolerance needed.
        assert torch.equal(output, expected.to(output.device)), \
            f"{name} differs from the independent padded chunk"
        # Rows at global positions >= rows are the zero padding rows.
        first_padding_row = max(rows - start, 0)
        assert torch.count_nonzero(output[first_padding_row:]) == 0, \
            f"{name} has non-zero padding rows"
        assert not torch._C._is_alias_of(output, source), \
            f"{name} aliases the input (official clones the no-pad chunk)"
        assert output.is_contiguous(), f"{name} output must be contiguous"

    assert torch.equal(ref_out, res_out), "reference and solution disagree"
    assert torch._C._is_alias_of(ref_out, ref_source) == \
        torch._C._is_alias_of(res_out, res_source), \
        "reference and solution disagree on input aliasing"


def compute_ref_performance(x, tp_size, tp_rank):
    return measure(
        functools.partial(compute_ref, tp_size=tp_size, tp_rank=tp_rank), {"x": x}
    )


def compute_res_performance(x, tp_size, tp_rank):
    return measure(
        functools.partial(compute_res, tp_size=tp_size, tp_rank=tp_rank), {"x": x}
    )


@label("v1_sequence_parallel_chunk")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sequence_parallel_chunk(config):
    shape, dtype, tp_size, tp_rank = config
    inputs = input_build(config)
    rows, hidden = math.prod(shape[:-1]), shape[-1]

    # Two equivalent inputs, one per side; both start from the same CPU tensor.
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    for side in (ref_inputs, res_inputs):
        assert side["x"].shape == (rows, hidden)
        assert side["x"].dtype == dtype
        assert side["x"].is_contiguous()

    ref_out = compute_ref(**ref_inputs, tp_size=tp_size, tp_rank=tp_rank)
    res_out = compute_res(**res_inputs, tp_size=tp_size, tp_rank=tp_rank)
    compare(ref_out, res_out, ref_inputs["x"], res_inputs["x"], config, dtype)

    # The operator never writes to its input.
    for side in (ref_inputs, res_inputs):
        torch.testing.assert_close(side["x"].cpu(), inputs["x"], rtol=0, atol=0)

    ref_time = compute_ref_performance(**inputs, tp_size=tp_size, tp_rank=tp_rank)
    res_time = compute_res_performance(**inputs, tp_size=tp_size, tp_rank=tp_rank)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time,
        speedup=ref_time / res_time if res_time else float("inf"),
    )
