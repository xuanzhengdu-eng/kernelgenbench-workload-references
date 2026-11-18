"""Accuracy and performance for v1_softmax_sglang (row 69, SGLang branch).

Contract (SGLang sampling softmax): ``temperatures`` carries one value per
request and each value covers ``rows_per_request`` consecutive rows; without an
``out`` buffer the result dtype follows torch promotion of
``logits / temperatures``, while with ``out`` the buffer is written in place and
returned (so its dtype wins, including ``out is logits``).
Official: https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/sampling/__init__.py#L67-L105
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows62_71_checks import (
    fresh, measure, result, unchanged,
)
from sandbox.verifier.test_parametrize import label, parametrize

# (rows, vocab, dtype, temperature_dtype, rows_per_request, out_mode)
COMBOS = [
    (rows, vocab, dtype, temp_dtype, rows_per_request, out_mode)
    for rows, vocab in [(8, 128), (6, 1000), (32, 1024)]
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
    for temp_dtype, rows_per_request in [(None, 1), (torch.float32, 1),
                                         (torch.float32, 2), (dtype, 1)]
    for out_mode in ("none", "fresh", "inplace")
]


def input_build(config):
    rows, vocab, dtype, temp_dtype, rows_per_request, out_mode = config
    generator = torch.Generator(device="cpu").manual_seed(6911)
    logits = torch.randn((rows, vocab), dtype=dtype, generator=generator)
    temperatures = None
    if temp_dtype is not None:
        requests = rows // rows_per_request
        # linspace is deterministic and has no generator argument upstream either.
        temperatures = torch.linspace(0.5, 1.5, requests, dtype=temp_dtype).view(-1, 1)
    return {"input": logits, "temperatures": temperatures,
            "rows_per_request": rows_per_request, "out_mode": out_mode}


def build_out(payload):
    """``inplace`` aliases the logits, ``fresh`` is a separate buffer, ``none`` skips it."""
    if payload["out_mode"] == "none":
        return None
    if payload["out_mode"] == "inplace":
        return payload["input"]
    return torch.empty_like(payload["input"])


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_softmax_sglang(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_softmax_sglang(**inputs)


def compare(ref_out, res_out):
    assert isinstance(res_out, torch.Tensor) and isinstance(ref_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    rtol, atol = (4e-6, 2e-6) if res_out.dtype == torch.float32 else (2e-3, 1e-3)
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)


@label("v1_softmax_sglang")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_softmax_sglang(config):
    rows, vocab, dtype, temp_dtype, rows_per_request, out_mode = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    ref_inputs["out"] = build_out(ref_inputs)
    res_inputs["out"] = build_out(res_inputs)
    ref_inputs.pop("out_mode")
    res_inputs.pop("out_mode")
    before_logits = ref_inputs["input"].clone()
    before_res_logits = res_inputs["input"].clone()
    assert ref_inputs["input"].shape == (rows, vocab) and ref_inputs["input"].dtype == dtype

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out)

    # Contract: without out the dtype follows torch promotion, with out the
    # buffer is written and returned, so the buffer dtype decides.
    if out_mode == "none":
        expected_dtype = torch.float32 if temp_dtype is not None and temp_dtype != dtype else dtype
        assert res_out.dtype == expected_dtype, (config, res_out.dtype, expected_dtype)
    else:
        assert res_out.dtype == dtype
        assert res_out.data_ptr() == res_inputs["out"].data_ptr()
        assert torch.equal(res_inputs["out"], res_out)
    if out_mode == "inplace":
        assert not torch.equal(res_inputs["input"], before_res_logits)
        assert not torch.equal(ref_inputs["input"], before_logits)
    else:
        assert torch.equal(res_inputs["input"], before_res_logits)
        assert torch.equal(ref_inputs["input"], before_logits)

    # Contract: rows sum to one.  The in-place buffer already holds the result,
    # so read the probabilities from the returned tensor.
    torch.testing.assert_close(res_out.float().sum(dim=-1),
                               torch.ones(rows, device=res_out.device),
                               rtol=0, atol=5e-3)

    # Contract: each request temperature covers consecutive rows, so scaling
    # with block-repeated values must give a different result.
    if temp_dtype is not None:
        temperatures = ref_inputs["temperatures"].float()
        scaled = before_res_logits.float() / torch.repeat_interleave(
            temperatures, rows_per_request, dim=0)
        torch.testing.assert_close(res_out.float(), torch.softmax(scaled, dim=-1),
                                   rtol=2e-3, atol=2e-3)
        if rows_per_request > 1:
            blockwise = torch.softmax(before_res_logits.float() / temperatures.repeat(
                rows_per_request, 1), dim=-1)
            assert not torch.allclose(res_out.float(), blockwise, rtol=1e-3, atol=1e-3), \
                "rows_per_request must group consecutive rows per request"

    # Timing: for the in-place variant the logits are restored before each call.
    def bench(side, payload):
        out_buffer = payload["out"]

        def step():
            if out_mode == "inplace":
                out_buffer.copy_(before_logits)
            payload["out"] = out_buffer
            side(**payload)

        return measure(step)

    original = before_logits
    ref_time = bench(kernelgenbench.baseline.v1_softmax_sglang, ref_inputs)
    res_time = bench(kernelgenbench.solution.v1_softmax_sglang, res_inputs)
    del original
    return result(ref_time, res_time)
