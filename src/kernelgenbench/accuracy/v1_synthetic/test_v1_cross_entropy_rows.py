"""Accuracy and performance for v1_cross_entropy_rows.

Operator: deepspeed.runtime.chunked_cross_entropy.chunked_cross_entropy with
reduction='none' (official chunked CE, Triton on CUDA via backend='auto').
Official source (pinned commit):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/runtime/chunked_cross_entropy.py#L159-L179
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/runtime/chunked_cross_entropy.py#L198-L239

Item: v1_cross_entropy_rows (723-row 168). ``compute_ref`` loads this file's
reference from ``kernelgenbench.baseline``; ``compute_res`` calls the real
official entry bound as ``kernelgenbench.solution``.

logits=(rows, vocab) FP32/FP16/BF16, target=(rows,) int64; output=(rows,) FP32.
Cases cover a partially ignored target, an all-ignored target (all zeros) and a
fully valid target. The old reference's label_smoothing branch has no upstream
parameter and was removed. Key assertions: shape/dtype, exact zeros on ignored
rows, fresh output, inputs untouched.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows162_171_checks as checks


COMBOS = [
    (rows, vocab, dtype, kind)
    for rows, vocab in [(7, 32), (19, 67)]
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
    for kind in ("partial", "all_ignored", "none")
]


def input_build(config):
    rows, vocab, dtype, kind = config
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    logits = torch.randn((rows, vocab), dtype=dtype, generator=generator)
    target = torch.randint(vocab, (rows,), dtype=torch.long, generator=generator)
    if kind == "all_ignored":
        target.fill_(-100)
    elif kind == "partial":
        target[::3] = -100
    return {"logits": logits, "target": target, "ignore_index": -100,
            "dtype": dtype, "kind": kind}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_cross_entropy_rows(
        logits=inputs["logits"], target=inputs["target"],
        ignore_index=inputs["ignore_index"])


def compute_res(**inputs):
    return kernelgenbench.solution.v1_cross_entropy_rows(
        logits=inputs["logits"], target=inputs["target"],
        ignore_index=inputs["ignore_index"])


def _copy(inputs):
    return {
        "logits": inputs["logits"].clone().to(kernelgenbench.device),
        "target": inputs["target"].clone().to(kernelgenbench.device),
        "ignore_index": inputs["ignore_index"],
        "dtype": inputs["dtype"],
        "kind": inputs["kind"],
    }


def compare(ref_out, res_out, dtype=None):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == torch.float32
    torch.testing.assert_close(res_out, ref_out, rtol=1e-5, atol=1e-5)


def compute_ref_performance(inputs):
    return checks.measure_factory(lambda: _copy(inputs), compute_ref,
                                  kernelgenbench.device)


def compute_res_performance(inputs):
    return checks.measure_factory(lambda: _copy(inputs), compute_res,
                                  kernelgenbench.device)


@label("v1_cross_entropy_rows")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_cross_entropy_rows(config):
    inputs = input_build(config)
    rows, vocab, dtype, kind = config
    ref_inputs = _copy(inputs)
    res_inputs = _copy(inputs)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)
    assert ref_out.shape == (rows,)

    ignored = ref_inputs["target"] == -100
    if ignored.any():
        assert torch.count_nonzero(ref_out[ignored]).item() == 0
        assert torch.count_nonzero(res_out[ignored]).item() == 0
    if kind == "all_ignored":
        assert torch.count_nonzero(ref_out).item() == 0
    if kind == "none":
        assert torch.isfinite(ref_out).all()
    for name in ("logits", "target"):
        torch.testing.assert_close(res_inputs[name].float(),
                                   inputs[name].float().to(res_inputs[name].device),
                                   rtol=0, atol=0)
    for produced in (ref_out, res_out):
        assert not torch._C._is_alias_of(produced, res_inputs["logits"])

    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return checks.timings(ref_time, res_time)
