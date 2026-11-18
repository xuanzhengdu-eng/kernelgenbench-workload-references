"""Accuracy and performance for v1_vllm_masked_moe_sum.

Contract: vLLM ``masked_moe_sum``. Unweighted FP32 accumulation of the slots
whose ``topk_ids`` entry is non-negative, written in place into ``output`` and
rounded once. Invalid slots are never read (NaN-safe) and every row is written,
so an all-invalid row becomes exactly zero.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import moe_rows222_231_checks as checks


# (tokens, topk, hidden, dtype, route pattern)
COMBOS = [
    (1, 3, 9, torch.bfloat16, "mixed"),
    (4, 2, 64, torch.float16, "invalid"),
    (17, 7, 1001, torch.bfloat16, "mixed"),
    (3, 4, 2057, torch.float32, "mixed"),
    (8, 5, 33, torch.float32, "valid"),
    (2, 3, 128, torch.float16, "cancel"),
    (0, 4, 16, torch.float32, "valid"),
]


def input_build(config):
    tokens, topk, hidden, dtype, pattern = config
    generator = torch.Generator(device="cpu").manual_seed(431 + tokens + topk + hidden)
    intermediate = torch.randn((tokens, topk, hidden), generator=generator).to(dtype)
    ids = (torch.arange(tokens * topk, dtype=torch.int32).reshape(tokens, topk) % 11)
    if pattern == "invalid":
        ids.fill_(-1)
    elif pattern == "mixed":
        ids.reshape(-1)[1::3] = -1
    if pattern in ("invalid", "mixed"):
        intermediate = torch.where(
            (ids < 0).unsqueeze(-1),
            torch.full_like(intermediate, float("nan")),
            intermediate,
        ).contiguous()
    if pattern == "cancel":
        # Sequential in-dtype accumulation distinguishes this from an FP32 sum.
        intermediate[:, 0].fill_(2048)
        intermediate[:, 1].fill_(1)
        intermediate[:, 2].fill_(-2048)
    output = torch.full((tokens, hidden), -999.0, dtype=dtype)
    return {"intermediate": intermediate, "topk_ids": ids, "output": output}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_vllm_masked_moe_sum(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_vllm_masked_moe_sum(**inputs)


def compare(ref_out, res_out, dtype=None):
    assert ref_out is None and res_out is None, "masked_moe_sum returns None"


def compute_ref_performance(**inputs):
    return checks.measure(kernelgenbench.baseline.v1_vllm_masked_moe_sum, inputs, device=kernelgenbench.device)


def compute_res_performance(**inputs):
    return checks.measure(kernelgenbench.solution.v1_vllm_masked_moe_sum, inputs, device=kernelgenbench.device)


@label("v1_vllm_masked_moe_sum")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_masked_moe_sum(config):
    tokens, topk, hidden, dtype, pattern = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    intermediate_before = res_inputs["intermediate"].cpu().clone()
    ids_before = res_inputs["topk_ids"].cpu().clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    # Output is written in place, every row exactly once, invalid slots skipped.
    expected = torch.zeros((tokens, hidden), dtype=dtype)
    for row in range(tokens):
        total = torch.zeros(hidden, dtype=torch.float32)
        for slot in range(topk):
            if int(res_inputs["topk_ids"][row, slot]) >= 0:
                total = total + inputs["intermediate"][row, slot].float()
        expected[row] = total.to(dtype)
    torch.testing.assert_close(res_inputs["output"].cpu(), expected, rtol=0, atol=0)
    if pattern == "invalid" and tokens:
        assert bool((res_inputs["output"] == 0).all())
    if pattern == "mixed":
        assert not torch.isnan(res_inputs["output"]).any(), "NaN from invalid slots leaked"
    torch.testing.assert_close(
        res_inputs["intermediate"].cpu(), intermediate_before, rtol=0, atol=0, equal_nan=True
    )
    assert torch.equal(res_inputs["topk_ids"].cpu(), ids_before)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return checks.timings(ref_time, res_time)
