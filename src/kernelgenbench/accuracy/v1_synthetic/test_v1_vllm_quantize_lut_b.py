"""Accuracy and performance for v1_vllm_quantize_lut_b.

Item: v1_vllm_quantize_lut_b (723-row 416).
Official source (pinned commit f358f057):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/lut_b_utils.py#L56-L200

Input: ``weight[E, N, K]`` (2D squeezes E) with N%8==0 and K%64==0.  The
function returns ``(packed, codebooks, scales)``; the 2D form squeezes the
expert dim.  ``num_iterations`` and ``max_tiles_per_chunk`` are keyword-only.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic import rows412_421_checks as checks
from sandbox.verifier.test_parametrize import label, parametrize


# experts, N, K, squeeze, dtype, iterations, chunk size, all-zero tile.
CASES = [
    (1, 8, 64, True, torch.float32, 2, 4096, False),
    (2, 16, 64, False, torch.bfloat16, 2, 1, False),
    (1, 8, 128, True, torch.float16, 0, 2, False),
    (1, 8, 64, True, torch.float32, 2, 1, True),
    (2, 16, 128, False, torch.float32, 1, 3, False),
]
COMBOS = CASES


def input_build(case):
    experts, n, k, squeeze, dtype, iterations, chunk, all_zero = case
    device = kernelgenbench.device
    weight = ((torch.arange(experts * n * k, dtype=torch.float32) % 33) - 16).reshape(
        experts, n, k
    ).to(dtype)
    if all_zero:
        weight.zero_()
    weight = weight.to(device)
    if squeeze:
        weight = weight[0]
    return dict(weight=weight, num_iterations=iterations, max_tiles_per_chunk=chunk)


def compute_ref(**kwargs):
    return kernelgenbench.baseline.v1_vllm_quantize_lut_b(**kwargs)


def compute_res(**kwargs):
    return kernelgenbench.solution.v1_vllm_quantize_lut_b(**kwargs)


def compare(ref_out, res_out):
    assert isinstance(ref_out, tuple) and len(ref_out) == 3
    assert isinstance(res_out, tuple) and len(res_out) == 3
    for expected, actual in zip(ref_out, res_out):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert torch.equal(expected.cpu().view(torch.uint8), actual.cpu().view(torch.uint8))


@label("v1_vllm_quantize_lut_b")
@parametrize("case", COMBOS)
@pytest.mark.parametrize("case", COMBOS)
def test_accuracy_v1_vllm_quantize_lut_b(case):
    inputs = input_build(case)
    snapshot = inputs["weight"].clone()
    ref_inputs = {"weight": inputs["weight"].clone(), "num_iterations": inputs["num_iterations"], "max_tiles_per_chunk": inputs["max_tiles_per_chunk"]}
    res_inputs = {"weight": inputs["weight"].clone(), "num_iterations": inputs["num_iterations"], "max_tiles_per_chunk": inputs["max_tiles_per_chunk"]}

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out)

    assert torch.equal(ref_inputs["weight"], snapshot), "reference modified weight"
    assert torch.equal(res_inputs["weight"], snapshot), "candidate modified weight"
    for produced in res_out:
        assert not torch._C._is_alias_of(produced, res_inputs["weight"])

    ref_time = checks.measure(compute_ref, lambda: input_build(case), kernelgenbench.device)
    res_time = checks.measure(compute_res, lambda: input_build(case), kernelgenbench.device)
    return checks.timings(ref_time, res_time)
