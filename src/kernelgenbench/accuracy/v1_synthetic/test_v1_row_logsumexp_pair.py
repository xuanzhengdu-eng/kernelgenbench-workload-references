"""Accuracy and performance for v1_row_logsumexp_pair (723 row 54).

Official entry: SGLang ``sglang.srt.layers.logsumexp.row_logsumexp`` (Triton
online row reduction, one program per row, fp32 accumulators).
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/logsumexp.py#L86-L110

Inputs: ``x`` is 2-D (rows, cols) in fp16/bf16/fp32. The four shapes cover
zero rows, zero columns, a normal row, and 16391 columns, which is larger than
the kernel's 16384-element block so the loop runs twice. Key assertions: the
result is a two-tensor tuple of fp32 vectors, the empty cases follow the
documented contract, the two outputs are distinct tensors, and ``x`` is not
modified.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic import rows52_61_checks as checks
from sandbox.verifier.test_parametrize import label, parametrize


SHAPES = [(0, 16), (3, 0), (8, 128), (2, 16391)]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
COMBOS = [(shape, dtype) for shape in SHAPES for dtype in DTYPES]


def input_build(config):
    shape, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(20261115)
    # A large common offset makes the (max, log_sum) split observable.
    base = 12.0 + torch.randn(shape, dtype=torch.float32, generator=generator)
    return {"x": base.to(dtype)}


def compute_ref(x):
    return kernelgenbench.baseline.v1_row_logsumexp_pair(x)


def compute_res(x):
    return kernelgenbench.solution.v1_row_logsumexp_pair(x)


def compare(ref_out, res_out, dtype):
    assert isinstance(ref_out, tuple) and len(ref_out) == 2, "reference must return (max, log_sum)"
    assert isinstance(res_out, tuple) and len(res_out) == 2, "solution must return (max, log_sum)"
    assert res_out[0] is not res_out[1], "the two output vectors must be distinct tensors"
    rows = ref_out[0].shape[0]
    for index, (ref, res) in enumerate(zip(ref_out, res_out)):
        assert res.shape == ref.shape == (rows,), (index, res.shape, ref.shape)
        assert res.dtype == ref.dtype == torch.float32, (index, res.dtype, ref.dtype)
        assert res.device == ref.device
        torch.testing.assert_close(res, ref, rtol=1e-5, atol=1e-5)


def compute_ref_performance(x):
    return checks.measure(kernelgenbench.baseline.v1_row_logsumexp_pair, {"x": x},
                          kernelgenbench.device)


def compute_res_performance(x):
    return checks.measure(kernelgenbench.solution.v1_row_logsumexp_pair, {"x": x},
                          kernelgenbench.device)


@label("v1_row_logsumexp_pair")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_row_logsumexp_pair(config):
    shape, dtype = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    assert res_inputs["x"].shape == shape and res_inputs["x"].dtype == dtype

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    rows, cols = shape
    if rows == 0:
        assert ref_out[0].numel() == res_out[0].numel() == 0
    elif cols == 0:
        assert torch.isinf(ref_out[0]).all() and (ref_out[0] < 0).all()
        assert torch.equal(ref_out[1], torch.zeros(rows, device=ref_out[1].device))
        assert torch.isinf(res_out[0]).all() and (res_out[0] < 0).all()
        assert torch.equal(res_out[1], torch.zeros(rows, device=res_out[1].device))
    else:
        # max + log(sum(exp(x - max))) reconstructs logsumexp(x).
        for out, side in ((ref_out, ref_inputs), (res_out, res_inputs)):
            reconstructed = out[0] + out[1]
            expected = torch.logsumexp(side["x"].float(), dim=-1)
            torch.testing.assert_close(reconstructed, expected, rtol=1e-5, atol=1e-5)
    for side in (ref_inputs, res_inputs):
        assert not any(torch._C._is_alias_of(out, side["x"]) for out in
                       (ref_out[0], ref_out[1], res_out[0], res_out[1]))
    torch.testing.assert_close(res_inputs["x"].cpu(), inputs["x"], rtol=0, atol=0)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return checks.timings(ref_time, res_time)
