"""Accuracy and performance for v1_vllm_moe_fused_mul_sum.

Contract: vLLM ``moe_fused_mul_sum``. Weighted FP32 accumulation over the top-k
slots, optional ``topk_ids``/``expert_map`` masking, optional pre-allocated
output, and a device ``num_valid_tokens`` cutoff. Invalid slots are not loaded
(NaN-safe), and all-invalid or out-of-range rows are left untouched.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import moe_rows222_231_checks as checks


# (tokens, topk, hidden, input dtype, weight dtype, ids, expert_map,
#  num_valid_tokens, preallocated output)
COMBOS = [
    (4, 3, 17, torch.float32, torch.float32, False, False, False, False),
    (4, 3, 17, torch.float16, torch.float32, True, False, False, True),
    (4, 3, 32, torch.float16, torch.float32, True, True, True, True),
    (5, 2, 257, torch.bfloat16, torch.bfloat16, True, True, False, True),
    (4, 3, 17, torch.float32, torch.float16, True, False, True, True),
    (0, 2, 8, torch.float16, torch.float32, False, False, False, True),
    (3, 1, 5, torch.bfloat16, torch.float32, True, True, False, False),
]


def input_build(config):
    tokens, topk, hidden, dtype, weight_dtype, use_ids, use_map, use_count, prealloc = config
    generator = torch.Generator(device="cpu").manual_seed(2280)
    inputs = torch.randn((tokens, topk, hidden), dtype=dtype, generator=generator)
    weights = torch.rand((tokens, topk), dtype=weight_dtype, generator=generator)
    ids = None
    expert_map = None
    if use_ids:
        ids = (torch.arange(tokens * topk, dtype=torch.int32).reshape(tokens, topk) % 4)
        if tokens:
            ids[0, :] = -1                      # all-invalid row: preserved
            inputs[0] = float("nan")
        if tokens > 1:
            ids[1, 0] = -1                      # one invalid slot: skipped, NaN safe
            inputs[1, 0] = float("nan")
        if use_map:
            expert_map = torch.tensor([0, 1, -1, 2], dtype=torch.int32)
            if tokens > 2:
                ids[2, 0] = 2                   # maps to -1: skipped, NaN safe
                inputs[2, 0] = float("nan")
    valid = None
    if use_count:
        valid = max(tokens - 1, 0)
        if tokens:
            inputs[valid] = float("nan")        # rows past num_valid_tokens: untouched
    outputs = torch.full((tokens, hidden), -77.0, dtype=dtype) if prealloc else None
    return {
        "inputs": inputs,
        "topk_weights": weights,
        "outputs": outputs,
        "topk_ids": ids,
        "expert_map": expert_map,
        "num_valid_tokens": None
        if valid is None
        else torch.tensor([valid], dtype=torch.int32),
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_vllm_moe_fused_mul_sum(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_vllm_moe_fused_mul_sum(**inputs)


def compare(ref_out, res_out, dtype=None):
    assert isinstance(ref_out, torch.Tensor) and isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape and res_out.dtype == ref_out.dtype == dtype
    rtol, atol = checks.tolerance(dtype)
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol, equal_nan=True)


def compute_ref_performance(**inputs):
    return checks.measure(kernelgenbench.baseline.v1_vllm_moe_fused_mul_sum, inputs, device=kernelgenbench.device)


def compute_res_performance(**inputs):
    return checks.measure(kernelgenbench.solution.v1_vllm_moe_fused_mul_sum, inputs, device=kernelgenbench.device)


@label("v1_vllm_moe_fused_mul_sum")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_moe_fused_mul_sum(config):
    tokens, topk, hidden, dtype, weight_dtype, use_ids, use_map, use_count, prealloc = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    sentinel = -77.0 if prealloc else None
    # ``outputs`` is the in-place destination and is expected to change.
    before = {
        name: value.cpu().clone()
        for name, value in res_inputs.items()
        if isinstance(value, torch.Tensor) and name != "outputs"
    }

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)
    if prealloc:
        assert res_out is res_inputs["outputs"], "pre-allocated output must be returned"

    # Rows that the kernel must not write stay at the sentinel; rows it must
    # write may never contain the NaN payload of a skipped slot.
    if prealloc and tokens:
        skip = set()
        if use_ids:
            skip.add(0)
        if use_count:
            skip.update(range(int(res_inputs["num_valid_tokens"].item()), tokens))
        for row in skip:
            assert torch.equal(
                res_out[row],
                torch.full_like(res_out[row], sentinel),
            ), f"row {row} should have been left untouched"
        written = [row for row in range(tokens) if row not in skip]
        if written:
            assert not torch.isnan(res_out[written]).any(), "NaN from skipped slots leaked"
    for name, value in before.items():
        torch.testing.assert_close(
            res_inputs[name].cpu(), value, rtol=0, atol=0, equal_nan=True
        )

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return checks.timings(ref_time, res_time)
