"""Accuracy and performance for v1_vocab_kl.

Operator: deepspeed.sequence.vocab_parallel_ops.vocab_parallel_kl_div forward
(native torch path, tp_group=None).
Official source (pinned commit):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/sequence/vocab_parallel_ops.py#L306-L335

Item: v1_vocab_kl (723-row 171). ``compute_ref`` loads this file's reference from
``kernelgenbench.baseline``; ``compute_res`` calls the real official entry bound as
``kernelgenbench.solution``.

student/teacher=(..., vocab) FP32/FP16/BF16; the result has shape (...,) in FP32.
Cases include an all -inf row (NaN on both sides), a token masked only in one
distribution (support mismatch) and both KL directions. The old reference
flattened the result to one dimension; the official contract preserves the
leading dimensions, so that shape is asserted here.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows162_171_checks as checks


CASES = [(2, 4, 32), (5, 67)]
COMBOS = [(case, dtype, reverse)
          for case in CASES
          for dtype in (torch.float32, torch.float16, torch.bfloat16)
          for reverse in (False, True)]

TOLERANCE = {
    torch.float32: (1e-5, 1e-5),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (4e-2, 4e-2),
}


def input_build(config):
    case, dtype, reverse = config
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    student = torch.randn(case, dtype=dtype, generator=generator)
    teacher = torch.randn(case, dtype=dtype, generator=generator)
    flat_s = student.view(-1, case[-1])
    flat_t = teacher.view(-1, case[-1])
    flat_s[0, :] = float('-inf')
    flat_t[0, :] = float('-inf')
    flat_t[1, 0] = float('-inf')
    flat_s[2, 1] = float('-inf')
    flat_t[2, 1] = float('-inf')
    return {"student": student, "teacher": teacher, "reverse": reverse, "dtype": dtype}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_vocab_kl(
        student=inputs["student"], teacher=inputs["teacher"],
        reverse=inputs["reverse"])


def compute_res(**inputs):
    return kernelgenbench.solution.v1_vocab_kl(
        student=inputs["student"], teacher=inputs["teacher"],
        reverse=inputs["reverse"])


def _copy(inputs):
    device = kernelgenbench.device
    return {"student": inputs["student"].clone().to(device),
            "teacher": inputs["teacher"].clone().to(device),
            "reverse": inputs["reverse"], "dtype": inputs["dtype"]}


def compare(ref_out, res_out, dtype=None):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == torch.float32
    rtol, atol = TOLERANCE[dtype]
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol, equal_nan=True)


def compute_ref_performance(inputs):
    return checks.measure_factory(lambda: _copy(inputs), compute_ref,
                                  kernelgenbench.device)


def compute_res_performance(inputs):
    return checks.measure_factory(lambda: _copy(inputs), compute_res,
                                  kernelgenbench.device)


@label("v1_vocab_kl")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vocab_kl(config):
    case, dtype, reverse = config
    inputs = input_build(config)
    ref_inputs = _copy(inputs)
    res_inputs = _copy(inputs)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    leading = case[:-1]
    assert ref_out.shape == leading
    assert torch.isnan(ref_out.reshape(-1)[0]), \
        "an all -inf row must be NaN"
    assert torch.isnan(ref_out.reshape(-1)[0]) == torch.isnan(res_out.reshape(-1)[0])
    if reverse:
        assert torch.isinf(ref_out).any(), \
            "reverse KL must be +inf where the teacher masks a token the student keeps"
        assert torch.isinf(res_out).sum().item() == torch.isinf(ref_out).sum().item()
    assert torch.count_nonzero(torch.isnan(ref_out)) == 1
    for name in ("student", "teacher"):
        torch.testing.assert_close(res_inputs[name], inputs[name].to(
            res_inputs[name].device), rtol=0, atol=0, equal_nan=True)

    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return checks.timings(ref_time, res_time)
