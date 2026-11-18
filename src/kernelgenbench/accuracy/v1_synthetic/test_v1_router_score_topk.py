"""Accuracy and performance for v1_router_score_topk (row 70, SGLang branch).

Contract (SGLang torch-native top-k): scoring is softmax / sigmoid /
sqrtsoftplus.  Without correction bias the scores are evaluated in fp32, the
selection is sorted and the ids are int64; with a bias the scores keep the
gating dtype, the ordering is unspecified, the selection uses score + bias and
the returned weight is the unbiased score.
Official: https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/moe/topk.py#L1157-L1204
"""
import pytest
import torch
import torch.nn.functional as F

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows62_71_checks import (
    assert_raises, fresh, measure, result, tolerance, unchanged,
)
from sandbox.verifier.test_parametrize import label, parametrize

# (topk, scoring_func, renormalize, correction_bias_present)
VARIANTS = [
    (1, "softmax", False, False),
    (2, "softmax", False, False),
    (4, "softmax", True, False),
    (2, "sigmoid", False, False),
    (2, "sigmoid", True, False),
    (4, "sigmoid", True, False),
    (2, "sqrtsoftplus", False, False),
    (4, "sqrtsoftplus", True, False),
    (2, "softmax", False, True),
    (4, "softmax", True, True),
    (2, "sigmoid", True, True),
    (4, "sqrtsoftplus", True, True),
]
# (tokens, hidden, experts, dtype)
CASES = [(32, 128, 16, torch.float32), (256, 1024, 16, torch.bfloat16),
         (7, 64, 8, torch.float32)]
COMBOS = [(tokens, hidden, experts, dtype) + variant
          for tokens, hidden, experts, dtype in CASES for variant in VARIANTS]


def input_build(config):
    tokens, hidden, experts, dtype, topk, scoring, renormalize, bias = config
    generator = torch.Generator(device="cpu").manual_seed(7001)
    hidden_states = torch.randn((tokens, hidden), dtype=dtype, generator=generator)
    gating = torch.randn((tokens, experts), dtype=dtype, generator=generator)
    correction_bias = None
    if bias:
        correction_bias = torch.randn((experts,), dtype=dtype, generator=generator) * 0.1
    return {"hidden_states": hidden_states, "gating_output": gating, "topk": topk,
            "renormalize": renormalize, "correction_bias": correction_bias,
            "scoring_func": scoring}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_router_score_topk(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_router_score_topk(**inputs)


def compare(ref_out, res_out, config):
    _, _, _, _, _, _, _, bias = config
    ref_w, ref_i = ref_out
    res_w, res_i = res_out
    assert res_w.shape == ref_w.shape and res_i.shape == ref_i.shape
    assert res_i.dtype == ref_i.dtype == torch.int64
    assert res_w.dtype == ref_w.dtype
    if bias:
        # The upstream selection is unsorted with a bias; the chosen set is the
        # contract, and the weights must match after aligning the columns.
        torch.testing.assert_close(res_i.sort(dim=-1).values, ref_i.sort(dim=-1).values)
        order = res_i.argsort(dim=-1)
        torch.testing.assert_close(res_w.gather(-1, order), ref_w.gather(-1, order),
                                   rtol=1e-2, atol=1e-2)
    else:
        rtol, atol = tolerance(torch.float32)
        torch.testing.assert_close(res_i, ref_i)
        torch.testing.assert_close(res_w, ref_w, rtol=rtol, atol=atol)


@label("v1_router_score_topk")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_router_score_topk(config):
    tokens, hidden, experts, dtype, topk, scoring, renormalize, bias = config
    inputs = input_build(config)
    before = {k: v.clone() for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    assert ref_inputs["gating_output"].shape == (tokens, experts)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config)
    unchanged(ref_inputs, before)
    unchanged(res_inputs, before)

    # Contract: fp32 scores without bias, gating dtype with a bias tensor.
    # Without a bias the scores are evaluated on the fp32 upcast; with a bias
    # the weights keep the gating dtype, except that renormalising divides by an
    # fp32 sum and therefore promotes the result back to fp32.
    expected_dtype = torch.float32 if (not bias or (renormalize and topk > 1)) else dtype
    assert res_out[0].dtype == expected_dtype, (config, res_out[0].dtype, expected_dtype)

    # Contract: one distinct expert per column and weights equal to the scores.
    assert (res_out[1].sort(dim=-1).values.diff(dim=-1) > 0).all()
    scores = {"softmax": lambda t: torch.softmax(t.float(), dim=-1),
              "sigmoid": lambda t: torch.sigmoid(t.float()),
              "sqrtsoftplus": lambda t: F.softplus(t.float()).sqrt()}[scoring]
    gathered = scores(res_inputs["gating_output"]).gather(1, res_out[1])
    if renormalize and topk > 1:
        gathered = gathered / (gathered.sum(-1, keepdim=True, dtype=torch.float32)
                               + 1e-20)
    torch.testing.assert_close(res_out[0].float(), gathered, rtol=2e-3, atol=2e-3)
    if renormalize and topk > 1:
        torch.testing.assert_close(res_out[0].float().sum(-1),
                                   torch.ones(tokens, device=res_out[0].device),
                                   rtol=0, atol=2e-3)

    # Contract: the unbiased path uses hidden_states only for the token count.
    if not bias:
        assert_raises(
            lambda: compute_ref(hidden_states=ref_inputs["hidden_states"][:-1],
                                gating_output=ref_inputs["gating_output"], topk=topk,
                                renormalize=renormalize, correction_bias=None,
                                scoring_func=scoring),
            "reference must reject a token-count mismatch")

    ref_time = measure(lambda: compute_ref(**ref_inputs))
    res_time = measure(lambda: compute_res(**res_inputs))
    return result(ref_time, res_time)
