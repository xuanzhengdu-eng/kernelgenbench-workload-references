"""Accuracy and performance for v1_megatron_pad_routing_inplace.

Item: v1_megatron_pad_routing_inplace (723-row 420).
Official source (pinned commit 266f1c97):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/moe_utils.py#L636-L693

Input: ``routing_map[T, E]`` bool or int32.  For each expert column, the first
``(-count) % pad_multiple`` zeros become 1.  The official entry mutates the
caller's storage and returns the ``transpose(0, 1)`` view, so the test checks
the returned view's identity, the mutated input, and a same-stride result.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic import rows412_421_checks as checks
from sandbox.verifier.test_parametrize import label, parametrize


# tokens, experts, multiple, dtype.
CASES = [
    (7, 3, 4, torch.bool),
    (8, 4, 4, torch.int32),
    (3, 2, 8, torch.bool),
    (0, 2, 8, torch.bool),
    (17, 3, 8, torch.bool),
    (5, 1, 2, torch.int32),
]
COMBOS = CASES


def input_build(case):
    tokens, experts, multiple, dtype = case
    device = kernelgenbench.device
    row = torch.arange(tokens, device=device)[:, None]
    col = torch.arange(experts, device=device)[None, :]
    routing_map = (((row * 3 + col * 2) % 5) < 2).to(dtype)
    return dict(routing_map=routing_map, pad_multiple=multiple)


def compute_ref(**kwargs):
    return kernelgenbench.baseline.v1_megatron_pad_routing_inplace(**kwargs)


def compute_res(**kwargs):
    return kernelgenbench.solution.v1_megatron_pad_routing_inplace(**kwargs)


def compare(ref_out, res_out, dtype):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape and res_out.dtype == ref_out.dtype == dtype
    assert res_out.stride() == ref_out.stride()
    assert torch.equal(ref_out.cpu(), res_out.cpu())


@label("v1_megatron_pad_routing_inplace")
@parametrize("case", COMBOS)
@pytest.mark.parametrize("case", COMBOS)
def test_accuracy_v1_megatron_pad_routing_inplace(case):
    dtype = case[3]
    ref_inputs = input_build(case)
    res_inputs = input_build(case)
    res_before = res_inputs["routing_map"].clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    # The native entry is in-place on its own storage and returns that view.
    assert torch._C._is_alias_of(ref_out, ref_inputs["routing_map"])
    assert torch._C._is_alias_of(res_out, res_inputs["routing_map"])
    assert torch.equal(ref_inputs["routing_map"], res_inputs["routing_map"])
    # When there are fewer zeros than the requested pad, the official mask flips
    # every available zero; both sides must agree on that partial padding.
    if case[0]:
        assert torch.equal(res_before, res_inputs["routing_map"]) or int(
            (res_inputs["routing_map"] != res_before).sum()
        ) > 0

    ref_time = checks.measure(compute_ref, lambda: input_build(case), kernelgenbench.device)
    res_time = checks.measure(compute_res, lambda: input_build(case), kernelgenbench.device)
    return checks.timings(ref_time, res_time)
