"""Accuracy and performance for v1_apply_min_p (Triton log-scale kernel).

Rows are mapped to min_p 0, 0.25 and 1.0. The zero rows must stay bit-identical
(the kernel returns before touching the row); the other rows keep only entries at
or above ``max(logits) + log(min_p)`` and set the rest to -inf. The reference
returns ``None`` and only mutates ``logits``.
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

SHAPES = [(6, 13), (6, 2050)]
COMBOS = [
    (shape, dtype, row_padding)
    for shape in SHAPES
    for dtype in (torch.float32, torch.bfloat16)
    for row_padding in (False, True)
]


def input_build(config):
    (rows, vocab), dtype, row_padding = config
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    base = torch.randn((rows, vocab + (8 if row_padding else 0)), generator=generator).to(dtype)
    logits = base[:, :vocab]
    return {
        "logits": logits,
        "expanded_idx_mapping": torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int64),
        "min_p": torch.tensor([0.0, 0.25, 1.0], dtype=torch.float32),
    }


def compute_ref(logits, **kwargs):
    return kernelgenbench.baseline.v1_apply_min_p(logits, **kwargs)


def compute_res(logits, **kwargs):
    return kernelgenbench.solution.v1_apply_min_p(logits, **kwargs)


def compute_ref_performance(initial, logits, **kwargs):
    return measure_inplace(compute_ref, {"logits": logits, **kwargs}, initial, logits.device)


def compute_res_performance(initial, logits, **kwargs):
    return measure_inplace(compute_res, {"logits": logits, **kwargs}, initial, logits.device)


@label("v1_apply_min_p")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_apply_min_p(config):
    (rows, vocab), dtype, row_padding = config
    inputs = input_build(config)
    device = kernelgenbench.device
    ref_inputs = move(inputs, device)
    res_inputs = move(inputs, device)
    initial = res_inputs["logits"].clone()
    min_p = res_inputs["min_p"].clone()
    mapping = res_inputs["expanded_idx_mapping"].clone()

    assert compute_ref(**ref_inputs) is None
    assert compute_res(**res_inputs) is None
    out = res_inputs["logits"]
    assert out.stride(0) == vocab + (8 if row_padding else 0)
    assert_unchanged(
        {k: v for k, v in res_inputs.items() if k != "logits"},
        {"min_p": min_p, "expanded_idx_mapping": mapping},
    )

    # min_p == 0 rows are untouched by the kernel early return.
    zero_rows = torch.where(min_p[mapping] == 0)[0]
    assert torch.equal(out[zero_rows], initial[zero_rows])
    # For the filtered rows, exactly the entries at/above max+log(min_p) survive.
    for row in torch.where(min_p[mapping] != 0)[0].tolist():
        p = float(min_p[mapping[row]])
        threshold = float(initial[row].float().max()) + torch.log(torch.tensor(p)).item()
        keep = initial[row].float() >= threshold
        assert torch.isneginf(out[row][~keep]).all()
        torch.testing.assert_close(
            out[row][keep].float(), initial[row][keep].float(), rtol=0, atol=0
        )
        assert bool(keep.any())

    rtol, atol = {
        torch.float32: (1e-6, 1e-6),
        torch.bfloat16: (1e-2, 8e-3),
    }[dtype]
    torch.testing.assert_close(
        out, ref_inputs["logits"], rtol=rtol, atol=atol, equal_nan=True
    )

    perf_ref = move(inputs, device)
    perf_res = move(inputs, device)
    ref_time = compute_ref_performance(perf_ref["logits"].clone(), **perf_ref)
    res_time = compute_res_performance(perf_res["logits"].clone(), **perf_res)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )
