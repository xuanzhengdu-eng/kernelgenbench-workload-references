"""Accuracy and performance for v1_megatron_pad_routing_fused.

Item: v1_megatron_pad_routing_fused (723-row 421).
Official source (pinned commit 266f1c97):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_pad_routing_map.py#L31-L110

Input: ``routing_map[T, E]`` bool or int32.  The official Triton entry is
``jit_fuser``-decorated, returns a fresh int32 ``[T, E]`` tensor (stride
``(1, T)``) for non-empty input, and leaves the input untouched.  For ``T == 0``
it returns the input object unchanged.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic import rows412_421_checks as checks
from sandbox.verifier.test_parametrize import label, parametrize


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
    return kernelgenbench.baseline.v1_megatron_pad_routing_fused(**kwargs)


def compute_res(**kwargs):
    return kernelgenbench.solution.v1_megatron_pad_routing_fused(**kwargs)


def compare(ref_out, res_out):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape and res_out.dtype == ref_out.dtype
    assert res_out.stride() == ref_out.stride()
    assert torch.equal(ref_out.cpu(), res_out.cpu())


@label("v1_megatron_pad_routing_fused")
@parametrize("case", COMBOS)
@pytest.mark.parametrize("case", COMBOS)
def test_accuracy_v1_megatron_pad_routing_fused(case):
    tokens, _, _, dtype = case
    ref_inputs = input_build(case)
    res_inputs = input_build(case)
    snapshot = res_inputs["routing_map"].clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out)

    if tokens:
        assert res_out.dtype == torch.int32
        assert not torch._C._is_alias_of(res_out, res_inputs["routing_map"])
        assert torch.equal(res_inputs["routing_map"], snapshot), "fused entry must not mutate the input"
    else:
        assert res_out is res_inputs["routing_map"]
        assert res_out.dtype == dtype

    ref_time = checks.measure(compute_ref, lambda: input_build(case), kernelgenbench.device)
    res_time = checks.measure(compute_res, lambda: input_build(case), kernelgenbench.device)
    return checks.timings(ref_time, res_time)
