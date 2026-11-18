"""Accuracy and performance for v1_repetition_penalty_ (vLLM routes).

Contiguous CUDA rows take vLLM's ``apply_repetition_penalties_cuda`` kernel;
padded (row-strided) rows take the torch route. Both must produce the same
result: positive logits divide by the penalty, non-positive logits multiply by
it, and tokens outside the prompt/output masks keep their exact value.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.sampling_test_utils import (
    assert_unchanged,
    measure_inplace,
    move,
)
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

SHAPES = [(3, 64), (3, 4096)]
COMBOS = [
    (shape, dtype, row_padding)
    for shape in SHAPES
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
    for row_padding in (False, True)
]


def input_build(config):
    (rows, vocab), dtype, row_padding = config
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    base = torch.randn((rows, vocab + (8 if row_padding else 0)), generator=generator).to(dtype)
    logits = base[:, :vocab]
    tokens = torch.arange(vocab).unsqueeze(0).expand(rows, -1)
    return {
        "logits": logits,
        "prompt_mask": tokens % 3 == 0,
        "output_mask": tokens % 5 == 0,
        # float32 penalties; the official CUDA op requires the logits dtype, so
        # the binding casts after moving to the device (vLLM always runs fp32
        # logits for this op).
        "repetition_penalties": torch.tensor([1.0, 2.0, 0.7], dtype=torch.float32),
    }


def compute_ref(logits, **kwargs):
    return kernelgenbench.baseline.v1_repetition_penalty_(logits, **kwargs)


def compute_res(logits, **kwargs):
    return kernelgenbench.solution.v1_repetition_penalty_(logits, **kwargs)


def compute_ref_performance(initial, logits, **kwargs):
    return measure_inplace(compute_ref, {"logits": logits, **kwargs}, initial, logits.device)


def compute_res_performance(initial, logits, **kwargs):
    return measure_inplace(compute_res, {"logits": logits, **kwargs}, initial, logits.device)


@label("v1_repetition_penalty_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_repetition_penalty_(config):
    (rows, vocab), dtype, row_padding = config
    inputs = input_build(config)
    device = kernelgenbench.device
    ref_inputs = move(inputs, device)
    res_inputs = move(inputs, device)
    penalties = res_inputs["repetition_penalties"].to(dtype)
    ref_inputs["repetition_penalties"] = penalties.clone()
    res_inputs["repetition_penalties"] = penalties.clone()
    initial = res_inputs["logits"].clone()
    masks = {
        "prompt_mask": res_inputs["prompt_mask"].clone(),
        "output_mask": res_inputs["output_mask"].clone(),
    }

    assert compute_ref(**ref_inputs) is None
    assert compute_res(**res_inputs) is None
    out = res_inputs["logits"]
    assert out.stride(0) == vocab + (8 if row_padding else 0)
    assert_unchanged(
        {k: v for k, v in res_inputs.items() if k != "logits"},
        {**masks, "repetition_penalties": penalties},
    )

    # The CUDA op rounds in the logits dtype (division), while the torch route
    # scales by the reciprocal, so allow the in-dtype rounding step.
    rtol, atol = {
        torch.float32: (1e-6, 1e-6),
        torch.float16: (4e-3, 4e-3),
        torch.bfloat16: (2e-2, 2e-2),
    }[dtype]
    torch.testing.assert_close(out, ref_inputs["logits"], rtol=rtol, atol=atol)

    # Hand-checked scaling on the masked entries.
    scaled = (masks["prompt_mask"] | masks["output_mask"])
    per_token = penalties.view(rows, 1).float().expand(rows, vocab)
    ratio = (out.float() / initial.float())[scaled]
    expected = torch.where(
        initial[scaled].float() > 0, 1.0 / per_token[scaled], per_token[scaled]
    )
    torch.testing.assert_close(ratio, expected, rtol=2e-3, atol=2e-3)
    # Zero logits stay exactly zero under either route.
    zeros = scaled & (initial == 0)
    if bool(zeros.any()):
        assert torch.equal(out[zeros], initial[zeros])
    # Unmasked entries are bit-identical.
    assert torch.equal(out[~scaled], initial[~scaled])

    perf_ref = move(inputs, device)
    perf_res = move(inputs, device)
    ref_time = compute_ref_performance(perf_ref["logits"].clone(), **perf_ref)
    res_time = compute_res_performance(perf_res["logits"].clone(), **perf_res)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )
