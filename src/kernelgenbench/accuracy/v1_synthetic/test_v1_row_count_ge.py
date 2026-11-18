"""Accuracy/performance for v1_row_count_ge (vLLM batched_count_greater_than).

``x`` is ``(batch, n)`` and ``values`` is ``(batch, 1)``; the result is int64 ``(batch,)``
counting elements ``>= values`` (ties included, matching the upstream computation even
though its docstring says "greater than").  ``n == 0`` returns zeros.
"""
import pytest
import torch
import triton.testing

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

COMBOS = [
    ((1, 0), 0, torch.float32),
    ((2, 8), 0, torch.float32),
    ((8, 1024), 0, torch.float32),
    ((4, 1024), 1, torch.float16),
    ((3, 7), -1, torch.bfloat16),
    ((2, 5), 0, torch.float16),
]


def input_build(config):
    """Integer draws with duplicates; ``offset`` moves the per-row threshold, 0 = ties."""
    shape, offset, dtype = config
    batch, width = shape
    generator = torch.Generator(device="cpu").manual_seed(20261115)
    x = torch.randint(-3, 4, shape, generator=generator).to(dtype)
    if width:
        values = x[:, :1] + offset
    else:
        values = torch.zeros((batch, 1), dtype=dtype)
    return {"x": x, "values": values}


def prepare(inputs, device):
    return {name: value.to(device) for name, value in inputs.items()}


def compute_ref(x, values):
    return kernelgenbench.baseline.v1_row_count_ge(x, values)


def compute_res(x, values):
    return kernelgenbench.solution.v1_row_count_ge(x, values)


def compare(ref_out, res_out, dtype=None):
    assert res_out.dtype == ref_out.dtype == torch.int64
    assert res_out.shape == ref_out.shape
    assert res_out.device == ref_out.device
    assert torch.equal(res_out, ref_out)


def compute_ref_performance(**inputs):
    """Median GPU time of one reference call; warmup and device sync are inside do_bench."""
    return triton.testing.do_bench(lambda: compute_ref(**inputs), warmup=25, rep=100)


def compute_res_performance(**inputs):
    """Same measurement for the solution side, identical inputs and conditions."""
    return triton.testing.do_bench(lambda: compute_res(**inputs), warmup=25, rep=100)


@label("v1_row_count_ge")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_row_count_ge(config):
    shape, offset, dtype = config
    inputs = input_build(config)
    ref_inputs = prepare(inputs, kernelgenbench.device)
    res_inputs = prepare(inputs, kernelgenbench.device)
    x_before = res_inputs["x"].clone()
    values_before = res_inputs["values"].clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    # Equality must be counted: every row has at least its own threshold element.
    batch, width = shape
    if width:
        expected = (inputs["x"].to(res_out.device) >=
                    inputs["values"].to(res_out.device)).sum(-1)
        assert torch.equal(res_out, expected)
        if offset <= 0:
            assert torch.all(res_out >= 1)
    else:
        assert torch.equal(res_out, torch.zeros((batch,), dtype=torch.int64,
                                                device=res_out.device))
    torch.testing.assert_close(res_inputs["x"], x_before, rtol=0, atol=0)
    torch.testing.assert_close(res_inputs["values"], values_before, rtol=0, atol=0)

    if config is COMBOS[0]:
        # The item keeps the documented (B,1) threshold shape; upstream broadcasting
        # would also accept (B,) (measured: same counts), so the guard is stated here.
        try:
            compute_ref(res_inputs["x"], res_inputs["values"].reshape(-1))
        except AssertionError:
            pass
        else:
            raise AssertionError("reference must require values of shape (batch, 1)")

    ref_time = compute_ref_performance(**ref_inputs)
    res_time = compute_res_performance(**res_inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
