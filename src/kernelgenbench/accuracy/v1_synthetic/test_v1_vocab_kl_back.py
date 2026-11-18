"""Accuracy and performance for v1_vocab_kl_back.

Operator: _VocabParallelKLDiv.backward of
deepspeed.sequence.vocab_parallel_ops (native torch path, tp_group=None).
Official source (pinned commit):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/sequence/vocab_parallel_ops.py#L337-L383

Item: v1_vocab_kl_back (723-row 170). ``compute_ref`` loads this file's reference
from ``kernelgenbench.baseline``; ``compute_res`` calls the real official backward
through ``vocab_parallel_kl_div`` autograd, bound as ``kernelgenbench.solution``.

student/teacher=(..., vocab) FP32/FP16/BF16; global_max/log_sum_exp=(2, rows)
FP32 saved statistics; kl=(rows,) FP32; grad_output=(rows,) FP32. Gradients have
the input shape and dtype, and either side may be omitted. Cases include both KL
directions, the three need_* combinations, undefined (all -inf) rows and a
support-mismatch row. Key assertions: tuple structure, the omitted gradient is
None on both sides, exact zeros on undefined rows, and values within the dtype
tolerance.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows162_171_checks as checks


CASES = [(2, 4, 32), (5, 67)]
NEEDS = [(True, True), (True, False), (False, True)]
COMBOS = [
    (case, dtype, reverse, need_student, need_teacher)
    for case in CASES
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
    for reverse in (False, True)
    for need_student, need_teacher in NEEDS
]

TOLERANCE = {
    torch.float32: (1e-5, 1e-5),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (4e-2, 4e-2),
}


def input_build(config):
    case, dtype, reverse, need_student, need_teacher = config
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
    stats = []
    for tensor in (student, teacher):
        rows = tensor.reshape(-1, case[-1]).float()
        maximum = rows.amax(dim=-1)
        log_sum_exp = torch.exp(rows - maximum[:, None]).sum(-1).log()
        stats.append((maximum, log_sum_exp))
    kl = kernelgenbench.baseline.v1_vocab_kl(student, teacher, reverse).reshape(-1)
    grad_output = torch.linspace(0.1, 1.0, kl.numel())
    return {
        "student": student,
        "teacher": teacher,
        "global_max": torch.stack([item[0] for item in stats]),
        "log_sum_exp": torch.stack([item[1] for item in stats]),
        "kl": kl,
        "grad_output": grad_output,
        "reverse": reverse,
        "need_student": need_student,
        "need_teacher": need_teacher,
        "dtype": dtype,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_vocab_kl_back(
        student=inputs["student"], teacher=inputs["teacher"],
        global_max=inputs["global_max"], log_sum_exp=inputs["log_sum_exp"],
        kl=inputs["kl"], grad_output=inputs["grad_output"],
        reverse=inputs["reverse"], need_student=inputs["need_student"],
        need_teacher=inputs["need_teacher"])


def compute_res(**inputs):
    return kernelgenbench.solution.v1_vocab_kl_back(
        student=inputs["student"], teacher=inputs["teacher"],
        global_max=inputs["global_max"], log_sum_exp=inputs["log_sum_exp"],
        kl=inputs["kl"], grad_output=inputs["grad_output"],
        reverse=inputs["reverse"], need_student=inputs["need_student"],
        need_teacher=inputs["need_teacher"])


def _copy(inputs):
    device = kernelgenbench.device
    return {
        "student": inputs["student"].clone().to(device),
        "teacher": inputs["teacher"].clone().to(device),
        "global_max": inputs["global_max"].clone().to(device),
        "log_sum_exp": inputs["log_sum_exp"].clone().to(device),
        "kl": inputs["kl"].clone().to(device),
        "grad_output": inputs["grad_output"].clone().to(device),
        "reverse": inputs["reverse"],
        "need_student": inputs["need_student"],
        "need_teacher": inputs["need_teacher"],
        "dtype": inputs["dtype"],
    }


def compare(ref_out, res_out, dtype=None):
    assert isinstance(ref_out, tuple) and len(ref_out) == 2
    assert isinstance(res_out, tuple) and len(res_out) == 2
    rtol, atol = TOLERANCE[dtype]
    for expected, actual in zip(ref_out, res_out):
        if expected is None:
            assert actual is None, "candidate returned a gradient the reference omits"
            continue
        assert actual is not None
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol,
                                   equal_nan=True)


def compute_ref_performance(inputs):
    return checks.measure_factory(lambda: _copy(inputs), compute_ref,
                                  kernelgenbench.device)


def compute_res_performance(inputs):
    return checks.measure_factory(lambda: _copy(inputs), compute_res,
                                  kernelgenbench.device)


@label("v1_vocab_kl_back")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vocab_kl_back(config):
    case, dtype, reverse, need_student, need_teacher = config
    inputs = input_build(config)
    ref_inputs = _copy(inputs)
    res_inputs = _copy(inputs)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    assert (ref_out[0] is None) == (not need_student)
    assert (ref_out[1] is None) == (not need_teacher)
    undefined = (ref_inputs["global_max"] == float('-inf')).any(dim=0)
    for output in (ref_out, res_out):
        for gradient, needed in ((output[0], need_student), (output[1], need_teacher)):
            if not needed:
                continue
            flat = gradient.reshape(-1, case[-1])
            if undefined.any():
                assert torch.count_nonzero(flat[undefined]).item() == 0, \
                    "undefined rows must get a zero gradient"

    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return checks.timings(ref_time, res_time)
