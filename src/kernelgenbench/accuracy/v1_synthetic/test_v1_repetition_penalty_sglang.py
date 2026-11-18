"""Accuracy and performance for v1_repetition_penalty_sglang.

SGLang's ``BatchedRepetitionPenalizer`` builds a dense ``cumulated_repetition_
penalties`` tensor (1.0 for tokens never emitted) and calls the torch.compile'd
``apply_scaling_penalties``. The contract is the same scaling rule as vLLM, but
the operand is the dense per-token scale instead of two masks, so the case
matrix drives the scale directly, including 1.0 rows.
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
CASES = ["prompt-output-masks", "dense-mixed"]
COMBOS = [
    (shape, dtype, case, row_padding)
    for shape in SHAPES
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
    for case in CASES
    for row_padding in (False, True)
]


def input_build(config):
    (rows, vocab), dtype, case, row_padding = config
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    base = torch.randn((rows, vocab + (8 if row_padding else 0)), generator=generator).to(dtype)
    logits = base[:, :vocab]
    tokens = torch.arange(vocab).unsqueeze(0).expand(rows, -1)
    if case == "prompt-output-masks":
        # Dense scale built the way the penalizer cumulates it, including 1.0.
        penalty_per_row = torch.tensor([1.0, 2.0, 0.7], dtype=torch.float32)
        seen = (tokens % 3 == 0) | (tokens % 5 == 0)
        scale = torch.where(seen, penalty_per_row.view(rows, 1), 1.0)
    else:
        # Every row uses a distinct positive scale so no entry stays at 1.0.
        scale = (1.0 + (tokens % 4) * 0.25).to(torch.float32)
        scale[0, ::7] = 3.0
    return {"logits": logits, "scaling_penalties": scale.contiguous()}


def compute_ref(logits, **kwargs):
    return kernelgenbench.baseline.v1_repetition_penalty_sglang(logits, **kwargs)


def compute_res(logits, **kwargs):
    return kernelgenbench.solution.v1_repetition_penalty_sglang(logits, **kwargs)


def compute_ref_performance(initial, logits, **kwargs):
    return measure_inplace(compute_ref, {"logits": logits, **kwargs}, initial, logits.device)


def compute_res_performance(initial, logits, **kwargs):
    return measure_inplace(compute_res, {"logits": logits, **kwargs}, initial, logits.device)


@label("v1_repetition_penalty_sglang")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_repetition_penalty_sglang(config):
    (rows, vocab), dtype, case, row_padding = config
    inputs = input_build(config)
    device = kernelgenbench.device
    ref_inputs = move(inputs, device)
    res_inputs = move(inputs, device)
    initial = res_inputs["logits"].clone()
    scale = res_inputs["scaling_penalties"].clone()

    assert compute_ref(**ref_inputs) is None
    assert compute_res(**res_inputs) is None
    out = res_inputs["logits"]
    assert out.stride(0) == vocab + (8 if row_padding else 0)
    assert_unchanged(
        {k: v for k, v in res_inputs.items() if k != "logits"},
        {"scaling_penalties": scale},
    )

    expected = torch.where(initial < 0, initial.float() * scale, initial.float() / scale)
    # ``out`` is rounded to the logits dtype, so compare in fp32 with the
    # dtype's own resolution.
    tol = {
        torch.float32: 1e-6,
        torch.float16: 2e-3,
        torch.bfloat16: 1e-2,
    }[dtype]
    torch.testing.assert_close(out.float(), expected, rtol=tol, atol=tol)
    assert torch.equal(out[scale == 1.0], initial[scale == 1.0])

    rtol, atol = {
        torch.float32: (1e-6, 1e-6),
        torch.float16: (2e-3, 1e-3),
        torch.bfloat16: (1e-2, 8e-3),
    }[dtype]
    torch.testing.assert_close(out, ref_inputs["logits"], rtol=rtol, atol=atol)

    perf_ref = move(inputs, device)
    perf_res = move(inputs, device)
    ref_time = compute_ref_performance(perf_ref["logits"].clone(), **perf_ref)
    res_time = compute_res_performance(perf_res["logits"].clone(), **perf_res)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )
