"""Accuracy and performance for v1_cross_entropy_rows_back.

Operator: _ChunkedCrossEntropy.backward of deepspeed.runtime.chunked_cross_entropy
(the official Triton/Torch chunked CE autograd path).
Official source (pinned commit):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/runtime/chunked_cross_entropy.py#L181-L195

Item: v1_cross_entropy_rows_back (723-row 167). ``compute_ref`` loads this file's
reference from ``kernelgenbench.baseline``; ``compute_res`` calls the real native
binding (official forward + autograd backward) bound as ``kernelgenbench.solution``.

logits=(rows, vocab) FP32/FP16/BF16, target=(rows,) int64; log_sum_exp and
grad_output are (rows,) FP32; the output has the logits shape and dtype. Cases
cover partially ignored rows (target[::3] = -100) and an all-ignored target.
Key assertions: shape/dtype, ignored rows get exactly zero gradient, all-ignored
yields an all-zero gradient, values match within the dtype tolerance, the output
is fresh, and inputs are untouched.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows162_171_checks as checks


COMBOS = [
    (rows, vocab, dtype, all_ignored)
    for rows, vocab in [(7, 32), (19, 67)]
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
    for all_ignored in (False, True)
]

TOLERANCE = {
    torch.float32: (1e-5, 1e-5),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (4e-2, 4e-2),
}


def input_build(config):
    rows, vocab, dtype, all_ignored = config
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    logits = torch.randn((rows, vocab), dtype=dtype, generator=generator)
    target = torch.randint(vocab, (rows,), dtype=torch.long, generator=generator)
    if all_ignored:
        target.fill_(-100)
    else:
        target[::3] = -100
    log_sum_exp = torch.logsumexp(logits.float(), dim=-1)
    grad_output = torch.linspace(0.1, 1.0, rows)
    return {
        "logits": logits,
        "target": target,
        "ignore_index": -100,
        "log_sum_exp": log_sum_exp,
        "grad_output": grad_output,
        "dtype": dtype,
        "all_ignored": all_ignored,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_cross_entropy_rows_back(
        logits=inputs["logits"], target=inputs["target"],
        ignore_index=inputs["ignore_index"], log_sum_exp=inputs["log_sum_exp"],
        grad_output=inputs["grad_output"])


def compute_res(**inputs):
    return kernelgenbench.solution.v1_cross_entropy_rows_back(
        logits=inputs["logits"], target=inputs["target"],
        ignore_index=inputs["ignore_index"], log_sum_exp=inputs["log_sum_exp"],
        grad_output=inputs["grad_output"])


def _copy(inputs):
    return {
        "logits": inputs["logits"].clone().to(kernelgenbench.device),
        "target": inputs["target"].clone().to(kernelgenbench.device),
        "ignore_index": inputs["ignore_index"],
        "log_sum_exp": inputs["log_sum_exp"].clone().to(kernelgenbench.device),
        "grad_output": inputs["grad_output"].clone().to(kernelgenbench.device),
        "dtype": inputs["dtype"],
        "all_ignored": inputs["all_ignored"],
    }


def compare(ref_out, res_out, dtype=None):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape, f"shape {tuple(res_out.shape)}"
    assert res_out.dtype == ref_out.dtype == dtype
    rtol, atol = TOLERANCE[dtype]
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)


def compute_ref_performance(inputs):
    return checks.measure_factory(lambda: _copy(inputs), compute_ref,
                                  kernelgenbench.device)


def compute_res_performance(inputs):
    return checks.measure_factory(lambda: _copy(inputs), compute_res,
                                  kernelgenbench.device)


@label("v1_cross_entropy_rows_back")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_cross_entropy_rows_back(config):
    inputs = input_build(config)
    rows, vocab, dtype, all_ignored = config
    ref_inputs = _copy(inputs)
    res_inputs = _copy(inputs)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)
    assert ref_out.shape == (rows, vocab)

    ignored = ref_inputs["target"] == -100
    if ignored.any():
        assert torch.count_nonzero(ref_out[ignored]).item() == 0
        assert torch.count_nonzero(res_out[ignored]).item() == 0
    if all_ignored:
        assert torch.count_nonzero(ref_out).item() == 0
        assert torch.count_nonzero(res_out).item() == 0
    for name in ("logits", "target", "log_sum_exp", "grad_output"):
        torch.testing.assert_close(res_inputs[name].float(),
                                   inputs[name].float().to(res_inputs[name].device),
                                   rtol=0, atol=0)
    for produced in (ref_out, res_out):
        assert not torch._C._is_alias_of(produced, res_inputs["logits"])

    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return checks.timings(ref_time, res_time)
