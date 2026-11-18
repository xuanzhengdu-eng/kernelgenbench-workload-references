"""Accuracy and performance for v1_apply_temperature_.

Rows are mapped to request temperatures 0, 1, 0.5 and 2.5. The 0 and 1 rows must
stay bit-identical (the Triton kernel returns before loading logits), while the
other rows are divided in fp32 and rounded back to the logits dtype. Row stride
padding and all three logits dtypes are covered.
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

SHAPES = [(4, 13), (4, 8193)]
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
    return {
        "logits": logits,
        "expanded_idx_mapping": torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        # 0 and 1 bypass the kernel, 0.5 scales up and 2.5 scales down.
        "temperature": torch.tensor([0.0, 1.0, 0.5, 2.5], dtype=torch.float32),
    }


def compute_ref(logits, **kwargs):
    return kernelgenbench.baseline.v1_apply_temperature_(logits, **kwargs)


def compute_res(logits, **kwargs):
    return kernelgenbench.solution.v1_apply_temperature_(logits, **kwargs)


def compare(ref_out, res_out, dtype):
    assert ref_out is None and res_out is None


def compute_ref_performance(initial, logits, **kwargs):
    return measure_inplace(compute_ref, {"logits": logits, **kwargs}, initial, logits.device)


def compute_res_performance(initial, logits, **kwargs):
    return measure_inplace(compute_res, {"logits": logits, **kwargs}, initial, logits.device)


@label("v1_apply_temperature_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_apply_temperature_(config):
    (rows, vocab), dtype, row_padding = config
    inputs = input_build(config)
    device = kernelgenbench.device
    ref_inputs = move(inputs, device)
    res_inputs = move(inputs, device)
    initial = res_inputs["logits"].clone()
    metadata = {k: v.clone() for k, v in res_inputs.items() if k != "logits"}

    assert compute_ref(**ref_inputs) is None
    assert compute_res(**res_inputs) is None
    out = res_inputs["logits"]
    assert out.stride(0) == vocab + (8 if row_padding else 0)
    assert_unchanged(
        {k: v for k, v in res_inputs.items() if k != "logits"}, metadata
    )

    # Bypass rows (temperature 0 and 1) must be bit-identical to the input.
    assert torch.equal(out[:2], initial[:2])
    # Scaled rows equal the fp32 division rounded back to the logits dtype.
    expected = (initial[2:].float() / torch.tensor([[0.5], [2.5]], device=device)).to(dtype)
    # fp32 division order can differ by one ulp from the Triton kernel.
    exact = dtype != torch.float32
    torch.testing.assert_close(
        out[2:], expected, rtol=0 if exact else 1e-6, atol=0 if exact else 1e-6
    )

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
