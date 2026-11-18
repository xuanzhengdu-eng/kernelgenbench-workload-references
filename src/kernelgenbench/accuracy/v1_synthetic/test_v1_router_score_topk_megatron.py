"""Accuracy and performance for v1_router_score_topk_megatron (row 70, Megatron branch).

Contract (Megatron eager routing): softmax with optional pre-softmax, sigmoid
with an optional expert bias used only for selection, optional group-limited
selection, optional scaling factor, and either dense [tokens, topk] output or a
sparse [tokens, experts] probability matrix plus a boolean routing map.
Official: https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/moe_utils.py#L671-L813
Grouped: https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/moe_utils.py#L578-L630
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows62_71_checks import (
    fresh, measure, result, tolerance, unchanged,
)
from sandbox.verifier.test_parametrize import label, parametrize

# (topk, score_function, use_pre_softmax, grouped, dense, scaling_factor, expert_bias)
VARIANTS = [
    (1, "softmax", False, False, True, None, False),
    (2, "softmax", False, False, True, None, False),
    (4, "softmax", True, True, True, 1.3, False),
    (2, "softmax", False, True, False, None, False),
    (4, "softmax", False, False, False, 1.3, False),
    (2, "softmax", True, False, False, 1.3, False),
    (1, "sigmoid", False, False, True, None, False),
    (2, "sigmoid", False, False, True, None, False),
    (4, "sigmoid", False, True, False, None, True),
    (2, "sigmoid", False, True, True, 1.3, True),
    (4, "sigmoid", False, False, False, None, False),
]
# (tokens, experts, dtype)
CASES = [(32, 16, torch.float32), (256, 16, torch.bfloat16), (7, 8, torch.float32)]
COMBOS = [(tokens, experts, dtype) + variant
          for tokens, experts, dtype in CASES for variant in VARIANTS]


def input_build(config):
    tokens, experts, dtype, topk, scoring, pre, grouped, dense, scaling, bias = config
    generator = torch.Generator(device="cpu").manual_seed(7011)
    logits = torch.randn((tokens, experts), dtype=torch.float32, generator=generator).to(dtype)
    expert_bias = None
    if bias:
        expert_bias = (torch.randn((experts,), generator=generator) * 0.1).to(dtype)
    return {"logits": logits, "topk": topk, "use_pre_softmax": pre,
            "num_groups": 4 if grouped else None, "group_topk": 2 if grouped else None,
            "scaling_factor": scaling, "score_function": scoring,
            "expert_bias": expert_bias, "dense_output": dense}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_router_score_topk_megatron(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_router_score_topk_megatron(**inputs)


def compare(ref_out, res_out, config):
    _, _, _, topk, _, _, _, dense, _, _ = config
    ref_p, ref_i = ref_out
    res_p, res_i = res_out
    if dense:
        assert res_p.shape == ref_p.shape and res_i.shape == ref_i.shape
        assert res_i.dtype == torch.int64
    else:
        assert res_p.shape == ref_p.shape and res_i.shape == ref_i.shape
        assert res_i.dtype == torch.bool
    torch.testing.assert_close(res_i, ref_i)
    torch.testing.assert_close(res_p.float(), ref_p.float(), rtol=4e-6, atol=2e-6)


def expected_scores(logits, ids, scoring, pre, bias_scores, topk, scaling):
    """Inline fp32 expectation for the probabilities at the selected experts."""
    logits_fp32 = logits.float()
    if scoring == "softmax":
        if pre:
            table = torch.softmax(logits_fp32, dim=-1)
            probs = table.gather(1, ids)
        else:
            probs = torch.softmax(logits_fp32.gather(1, ids), dim=-1)
    else:
        table = torch.sigmoid(logits_fp32)
        probs = table.gather(1, ids)
        if topk > 1:
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-20)
    if scaling:
        probs = probs * scaling
    return probs


@label("v1_router_score_topk_megatron")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_router_score_topk_megatron(config):
    tokens, experts, dtype, topk, scoring, pre, grouped, dense, scaling, bias = config
    inputs = input_build(config)
    before = {k: v.clone() for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    assert ref_inputs["logits"].dtype == dtype

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config)
    unchanged(ref_inputs, before)
    unchanged(res_inputs, before)

    probs = res_out[0]
    if dense:
        ids = res_out[1]
        assert (ids.sort(dim=-1).values.diff(dim=-1) > 0).all()
    else:
        assert res_out[1].sum(dim=-1).eq(topk).all()
        assert torch.all((probs == 0) | res_out[1])
        # The sparse map keeps index order; recover the ids and align both
        # tensors to ascending index order before the value comparison.
        ids = res_out[1].int().argsort(dim=-1)[:, -topk:]
        ids = ids.sort(dim=-1).values
        probs = probs.gather(1, ids)

    # Contract: the probability at each selected expert equals softmax over the
    # selected logits (or the gathered full softmax / renormalized sigmoid),
    # times the optional scaling factor.
    expected = expected_scores(res_inputs["logits"], ids, scoring, pre, bias, topk, scaling)
    rtol, atol = tolerance(dtype)
    torch.testing.assert_close(probs.float(), expected, rtol=rtol, atol=atol)

    # Contract: softmax without pre-softmax normalizes over the selected set,
    # pre-softmax keeps the full-distribution mass, so their sums differ.
    if scoring == "softmax" and topk > 1:
        total = probs.float().sum(-1) / (scaling or 1.0)
        rtol, atol = tolerance(dtype)
        if pre:
            assert (total < 1.0 + atol).all(), "pre-softmax keeps the full-distribution mass"
        else:
            torch.testing.assert_close(total, torch.ones_like(total), rtol=rtol, atol=atol)

    # Contract: group-limited selection keeps every chosen expert inside the
    # high-scoring groups, so a forbidden expert can never be selected.
    if grouped:
        # Official group scores come from the same score tensor that the routing
        # uses: raw logits (softmax without pre-softmax), the dtype-cast softmax
        # (with pre-softmax), or the dtype-cast sigmoid plus the expert bias.
        if scoring == "softmax":
            table = (ref_inputs["logits"] if not pre
                     else torch.softmax(ref_inputs["logits"].float(), dim=-1).to(dtype))
        else:
            table = torch.sigmoid(ref_inputs["logits"].float()).to(dtype)
            if bias:
                table = table + ref_inputs["expert_bias"]
        group_scores = table.view(tokens, 4, -1).topk(max(topk // 2, 1), dim=-1)[0].sum(-1)
        chosen_groups = group_scores.topk(2, dim=-1, sorted=False).indices
        allowed = torch.zeros((tokens, 4), dtype=torch.bool, device=probs.device)
        allowed.scatter_(1, chosen_groups, True)
        allowed = allowed.view(tokens, 4, 1).expand(tokens, 4, experts // 4).reshape(
            tokens, experts)
        chosen = torch.zeros((tokens, experts), dtype=torch.bool, device=probs.device)
        chosen.scatter_(1, ids, True)
        assert not (chosen & ~allowed).any(), \
            "group-limited routing chose an expert outside its selected groups"

    ref_time = measure(lambda: compute_ref(**ref_inputs))
    res_time = measure(lambda: compute_res(**res_inputs))
    return result(ref_time, res_time)
