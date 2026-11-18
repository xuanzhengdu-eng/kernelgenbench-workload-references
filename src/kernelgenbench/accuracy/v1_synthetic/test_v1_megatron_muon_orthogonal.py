"""Megatron TensorParallelMuon.orthogonalize, single rank, FP32 / highest.

The pinned Emerging-Optimizers v0.2.0 quintic Newton-Schulz iteration normalizes by
the Frobenius norm (floor 1e-7), applies five coefficient triples cyclically, and
transposes so the smaller dimension is orthogonalized.  Muon then scales by
``max(rows, cols) ** 0.5`` (spectral), ``(rows/cols) ** 0.5`` (unit_rms_norm) or
``max(1, rows/cols) ** 0.5`` (shape_scaling) and by ``extra_scale``.

With ``qkv`` the Megatron wrapper splits the rows into query groups, orthogonalizes
each Q/K/V block separately and concatenates them back.  Everything runs with
``torch.set_float32_matmul_precision('highest')``; the output is a new tensor and the
gradient is untouched.
"""
import pytest
import torch

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

# rows, columns, qkv split or None, steps, scale mode, extra scale, zero input
COMBOS = [
    (3, 5, None, 1, "spectral", 1.0, False),
    (5, 3, None, 5, "shape_scaling", 0.5, False),
    (4, 4, None, 7, "unit_rms_norm", 2.0, False),
    (8, 5, (2, 1, 1), 3, "spectral", 1.0, False),
    (12, 4, (2, 1, 1), 5, "unit_rms_norm", 0.75, False),
    (8, 7, (2, 1, 1), 2, "shape_scaling", 1.2, False),
    (4, 6, None, 5, "spectral", 1.0, True),
]
# Query-group splitting requires rows divisible by the split sum.
COMBOS = [case for case in COMBOS if case[2] is None or case[0] % sum(case[2]) == 0]


def input_build(case):
    rows, cols, qkv, steps, scale, extra, zero = case
    generator = torch.Generator(device="cpu").manual_seed(20261115)
    grad = torch.randn((rows, cols), dtype=torch.float32, generator=generator)
    if zero:
        grad.zero_()
    return {"grad": grad, "qkv": qkv, "steps": steps, "scale_mode": scale, "extra_scale": extra}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_megatron_muon_orthogonal(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_megatron_muon_orthogonal(**inputs)


def compare(ref_out, res_out, grad):
    assert isinstance(ref_out, torch.Tensor) and isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape == grad.shape
    assert res_out.dtype == ref_out.dtype == torch.float32
    assert res_out.device == ref_out.device
    torch.testing.assert_close(res_out, ref_out, rtol=1e-4, atol=1e-4)
    assert not torch._C._is_alias_of(res_out, grad), "orthogonalized update is a new tensor"


def _bench(call, kwargs):
    for _ in range(2):
        call(**kwargs)
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call(**kwargs)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return sorted(samples)[len(samples) // 2]


@label("v1_megatron_muon_orthogonal")
@parametrize("case", COMBOS)
@pytest.mark.parametrize("case", COMBOS)
def test_accuracy_v1_megatron_muon_orthogonal(case):
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        inputs = input_build(case)
        device = torch.device("cuda")
        ref_inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                      for key, value in inputs.items()}
        res_inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                      for key, value in inputs.items()}
        saved = res_inputs["grad"].clone()
        ref_out = compute_ref(**ref_inputs)
        res_out = compute_res(**res_inputs)
        compare(ref_out, res_out, ref_inputs["grad"])
        assert torch.equal(res_inputs["grad"], saved), "gradient was modified"
        assert torch.equal(ref_inputs["grad"], saved)

        timed = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                 for key, value in inputs.items()}
        ref_time = _bench(compute_ref, timed)
        res_time = _bench(compute_res, timed)
        return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                     speedup=ref_time / res_time if res_time else float("inf"))
    finally:
        torch.set_float32_matmul_precision(previous)
