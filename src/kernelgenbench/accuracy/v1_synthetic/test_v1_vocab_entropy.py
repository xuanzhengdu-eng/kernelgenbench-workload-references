"""Accuracy and performance for v1_vocab_entropy (DeepSpeed vocabulary entropy).

Input is ``(batch, time, vocab)`` or ``(rows, vocab)`` in FP32/FP16/BF16.
Row 0 is all ``-inf`` and row 1 has one ``-inf`` entry, so the test covers the
two documented boundaries: an undefined row returns NaN and a masked logit
contributes zero.  The output is ``(rows,)`` FP32.

``COMBOS``/``get_input`` keep the historical ``(case, dtype)`` shape because
the backward test for the same source reuses them.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.bias_activation_test_utils import fresh, measure
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


COMBOS = [
    (case, dtype)
    for case in [(2, 4, 32), (5, 67)]
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
]


def get_input(case, dtype, device="cpu"):
    generator = torch.Generator(device=device).manual_seed(20261109)
    logits = torch.randn(case, dtype=dtype, device=device, generator=generator)
    rows = logits.view(-1, case[-1])
    rows[0, :] = float("-inf")   # undefined distribution -> NaN entropy
    rows[1, 0] = float("-inf")   # masked logit -> zero contribution
    return {"logits": logits}


def input_build(config):
    case, dtype = config
    return get_input(case, dtype, device="cpu")


def compute_ref(logits):
    return kernelgenbench.baseline.v1_vocab_entropy(logits)


def compute_res(logits):
    return kernelgenbench.solution.v1_vocab_entropy(logits)


def compare(ref_out, res_out, dtype):
    assert isinstance(ref_out, torch.Tensor) and isinstance(res_out, torch.Tensor)
    assert ref_out.shape == res_out.shape
    assert ref_out.dtype == res_out.dtype == torch.float32
    rtol = {torch.float32: 1e-5, torch.float16: 5e-3, torch.bfloat16: 2e-2}[dtype]
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=rtol, equal_nan=True)


def compute_ref_performance(logits):
    return measure(compute_ref, {"logits": logits})


def compute_res_performance(logits):
    return measure(compute_res, {"logits": logits})


@label("v1_vocab_entropy")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vocab_entropy(config):
    case, dtype = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    for side in (ref_inputs, res_inputs):
        assert side["logits"].shape == case

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)
    # Per-token output keeps the leading shape of the logits.
    for output in (ref_out, res_out):
        assert output.shape == case[:-1]

    expected = inputs["logits"].to(kernelgenbench.device)
    for side in (ref_inputs, res_inputs):
        torch.testing.assert_close(side["logits"], expected, rtol=0, atol=0, equal_nan=True)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
