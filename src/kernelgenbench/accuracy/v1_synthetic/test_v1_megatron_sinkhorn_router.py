"""Accuracy and performance for v1_megatron_sinkhorn_router (row 71).

Contract: Megatron TopKRouter.sinkhorn_load_balancing.  training routes on the
sinkhorn-scaled cost in fp32 under no_grad, eval routes on the activation scores;
topk=1 uses sigmoid and topk>1 uses fp32 softmax cast to the logits dtype.  The
result is (activation * routing_map, routing_map) with a bool map, and the logits
are not modified.  moe_aux_loss_coeff must be 0.
Official: https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/router.py#L230-L262
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows62_71_checks import (
    fresh, measure, result, tolerance, unchanged,
)
from sandbox.verifier.test_parametrize import label, parametrize

# (tokens, experts, topk, training, dtype); topk <= experts.
COMBOS = [
    (tokens, experts, topk, training, dtype)
    for tokens, experts in [(5, 4), (64, 16), (7, 5)]
    for topk in (1, 2, 3)
    for training in (True, False)
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
    if topk <= experts
]


def input_build(config):
    tokens, experts, topk, training, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(2341 + tokens * 100 + experts)
    logits = torch.randn((tokens, experts), dtype=torch.float32, generator=generator).to(dtype)
    return {"logits": logits, "topk": topk, "training": training}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_megatron_sinkhorn_router(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_megatron_sinkhorn_router(**inputs)


def compare(ref_out, res_out, config):
    dtype = config[-1]
    ref_scores, ref_map = ref_out
    res_scores, res_map = res_out
    assert res_scores.shape == ref_scores.shape and res_map.shape == ref_map.shape
    assert res_scores.dtype == ref_scores.dtype == dtype
    assert res_map.dtype == ref_map.dtype == torch.bool
    rtol, atol = tolerance(dtype)
    torch.testing.assert_close(res_map, ref_map)
    torch.testing.assert_close(res_scores, ref_scores, rtol=rtol, atol=atol)


@label("v1_megatron_sinkhorn_router")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_sinkhorn_router(config):
    tokens, experts, topk, training, dtype = config
    inputs = input_build(config)
    before = {k: v.clone() for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    assert ref_inputs["logits"].shape == (tokens, experts)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config)
    unchanged(ref_inputs, before)
    unchanged(res_inputs, before)

    scores, routing = res_out
    # Contract: exactly topk experts per token, and scores are the activation
    # masked by the routing map (so unselected experts are exactly zero).
    assert routing.sum(dim=-1).eq(topk).all()
    assert torch.all((scores == 0) | routing)
    logits_fp32 = res_inputs["logits"].float()
    activation = torch.sigmoid(logits_fp32) if topk == 1 \
        else torch.softmax(logits_fp32, dim=-1).to(dtype).float()
    torch.testing.assert_close(scores.float(), activation * routing.float(),
                               rtol=tolerance(dtype)[0], atol=tolerance(dtype)[1])
    # Contract: topk=1 routes on sigmoid scores, not softmax.
    if topk == 1 and experts > 1:
        assert (scores.float() < 1.0).all()

    ref_time = measure(lambda: compute_ref(**ref_inputs))
    res_time = measure(lambda: compute_res(**res_inputs))
    return result(ref_time, res_time)
