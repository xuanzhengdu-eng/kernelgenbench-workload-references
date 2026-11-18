"""Accuracy and performance for v1_indices_to_multihot_back (Megatron backward).

Official entry: NVIDIA Megatron-LM ``IndicesToMultihot.backward``, pinned
commit 266f1c97dca477ca8b92c16087506da2000b0b84, source lines 239-280.  The
candidate binding calls that static autograd backward with the saved
position-map context, so the official ``_multihot_to_indices_kernel`` runs.

Input: ``grad_probs_in_multihot`` [tokens, num_local_experts], the int32
``position_map`` saved by the forward pass (unique slot per row, -1 absent),
and the forward ``topk``.  Returns the autograd four-tuple
(None, grad_probs [tokens, topk], None, None); each valid position gathers its
multihot gradient and unset slots stay 0.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic.rows202_211_test_utils import fresh, measure, result


# tokens, num_local_experts, top_k, dtype
COMBOS = [
    (tokens, experts, top_k, dtype)
    for tokens in (1, 4, 17)
    for experts in (4, 8)
    for top_k in (2, 3)
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
]


def input_build(config):
    """Pick unique experts per row and derive the forward position map."""
    tokens, experts, top_k, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(20261109 + tokens * 31 + experts)
    indices = torch.stack([
        torch.randperm(experts, generator=generator)[:top_k] for _ in range(tokens)
    ]).to(torch.int64)
    position_map = torch.full((tokens, experts), -1, dtype=torch.int32)
    for row in range(tokens):
        for slot in range(top_k):
            position_map[row, int(indices[row, slot])] = slot
    grad = torch.randn((tokens, experts), generator=generator, dtype=dtype)
    return {"grad_probs_in_multihot": grad, "position_map": position_map, "topk": top_k}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_indices_to_multihot_back(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_indices_to_multihot_back(**inputs)


def compute_ref_performance(inputs):
    """Time the reference on independently rebuilt inputs."""
    return measure(compute_ref, lambda: fresh(inputs, kernelgenbench.device))


def compute_res_performance(inputs):
    """Time the candidate on independently rebuilt inputs."""
    return measure(compute_res, lambda: fresh(inputs, kernelgenbench.device))


@label("v1_indices_to_multihot_back")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_indices_to_multihot_back(config):
    tokens, experts, top_k, dtype = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    before_grad = res_inputs["grad_probs_in_multihot"].clone()
    before_map = res_inputs["position_map"].clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)

    assert isinstance(ref_out, tuple) and len(ref_out) == 4
    assert isinstance(res_out, tuple) and len(res_out) == 4
    for index in (0, 2, 3):
        assert res_out[index] is None
    assert isinstance(res_out[1], torch.Tensor)
    assert res_out[1].shape == (tokens, top_k)
    assert res_out[1].dtype == dtype
    assert torch.equal(res_out[1], ref_out[1])

    # Independent gather: every valid map entry writes its multihot gradient
    # into the stored top-k slot; absent entries leave zeros.
    expected = torch.zeros((tokens, top_k), dtype=dtype, device=res_out[1].device)
    grad = res_inputs["grad_probs_in_multihot"]
    for row in range(tokens):
        for expert in range(experts):
            slot = int(res_inputs["position_map"][row, expert])
            if slot >= 0:
                expected[row, slot] = grad[row, expert]
    assert torch.equal(res_out[1], expected), "position-map gather contract"
    assert torch.equal(res_inputs["grad_probs_in_multihot"], before_grad)
    assert torch.equal(res_inputs["position_map"], before_map)

    ref_ms = compute_ref_performance(inputs)
    res_ms = compute_res_performance(inputs)
    return result(ref_ms, res_ms)
