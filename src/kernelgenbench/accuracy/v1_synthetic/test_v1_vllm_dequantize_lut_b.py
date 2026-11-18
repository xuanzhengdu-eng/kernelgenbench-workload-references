"""Accuracy and performance for v1_vllm_dequantize_lut_b.

Item: v1_vllm_dequantize_lut_b (723-row 415).
Official source (pinned commit f358f057):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/lut_b_utils.py#L203-L258

Input: ``packed[E, N/8, K/64, 192]`` uint8, ``codebooks[..., 8]`` E4M3, and
``scales[E, N/8, K/64]`` uint8 UE8M0.  The 2D form squeezes the expert dim.
The output is a fresh ``[N, K]`` (or ``[E, N, K]``) tensor.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic import rows412_421_checks as checks
from sandbox.verifier.test_parametrize import label, parametrize


# experts, N tiles, K tiles, squeeze expert dim, output dtype.
CASES = [
    (1, 1, 1, True, torch.float32),
    (2, 2, 1, False, torch.bfloat16),
    (1, 1, 2, True, torch.float16),
    (2, 1, 2, False, torch.float32),
    (2, 2, 2, False, torch.bfloat16),
]
COMBOS = CASES


def input_build(case):
    experts, n_tiles, k_tiles, squeeze, dtype = case
    device = kernelgenbench.device
    packed = (
        torch.arange(experts * n_tiles * k_tiles * 192, dtype=torch.int32) * 37 + 11
    ).to(torch.uint8).reshape(experts, n_tiles, k_tiles, 192).to(device)
    books = (
        (torch.arange(experts * n_tiles * k_tiles * 8, dtype=torch.float32) % 15) - 7
    ).reshape(experts, n_tiles, k_tiles, 8).to(torch.float8_e4m3fn).to(device)
    scales = (
        (torch.arange(experts * n_tiles * k_tiles, dtype=torch.int32) % 5 + 125)
        .to(torch.uint8)
        .reshape(experts, n_tiles, k_tiles)
        .to(device)
    )
    if squeeze:
        packed, books, scales = packed[0], books[0], scales[0]
    return dict(packed=packed, codebooks=books, scales=scales, out_dtype=dtype)


def compute_ref(**kwargs):
    return kernelgenbench.baseline.v1_vllm_dequantize_lut_b(**kwargs)


def compute_res(**kwargs):
    return kernelgenbench.solution.v1_vllm_dequantize_lut_b(**kwargs)


def compare(ref_out, res_out, dtype):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape and res_out.dtype == ref_out.dtype == dtype
    assert torch.equal(ref_out.cpu(), res_out.cpu())


@label("v1_vllm_dequantize_lut_b")
@parametrize("case", COMBOS)
@pytest.mark.parametrize("case", COMBOS)
def test_accuracy_v1_vllm_dequantize_lut_b(case):
    dtype = case[4]
    inputs = input_build(case)
    ref_inputs = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    res_inputs = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    snapshot = {k: v.clone() for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    for name in ("packed", "codebooks", "scales"):
        assert torch.equal(ref_inputs[name], snapshot[name]), f"reference modified {name}"
        assert torch.equal(res_inputs[name], snapshot[name]), f"candidate modified {name}"
        assert not torch._C._is_alias_of(res_out, res_inputs[name]), f"output aliases {name}"

    ref_time = checks.measure(compute_ref, lambda: input_build(case), kernelgenbench.device)
    res_time = checks.measure(compute_res, lambda: input_build(case), kernelgenbench.device)
    return checks.timings(ref_time, res_time)
