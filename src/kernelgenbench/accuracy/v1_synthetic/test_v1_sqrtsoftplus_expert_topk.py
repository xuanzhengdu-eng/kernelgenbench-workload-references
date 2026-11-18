"""Accuracy and performance for v1_sqrtsoftplus_expert_topk.

Official entry: ``sglang.srt.layers.moe.topk.fused_topk_torch_native`` with
``scoring_func="sqrtsoftplus"`` (SGLang 62384de1ae5e3bbd3159afd1639525b0bf39fbf9,
topk.py L1157-L1205).  Scores are ``sqrt(softplus(x))``; the no-bias path
upcasts to FP32, sorts top-k and returns int64 ids.  With a bias the gating
dtype is kept for scores, ids are unsorted and weights are gathered from the
unbiased scores.  Renormalization uses the FP32 sum plus 1e-20.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.moe_rows192_201_utils import (
    fresh,
    measure,
    timings,
)
from sandbox.verifier.test_parametrize import label, parametrize


# (tokens, experts, topk, use correction bias, renormalize, dtype)
COMBOS = [
    (0, 8, 2, False, False, torch.float32),
    (5, 8, 2, False, False, torch.bfloat16),
    (4, 8, 3, True, True, torch.float32),
    (1, 16, 4, False, True, torch.float16),
    (3, 16, 2, True, False, torch.bfloat16),
]


def input_build(config):
    tokens, experts, topk, use_bias, renormalize, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(195)
    hidden_states = torch.randn((tokens, 8), dtype=dtype, generator=generator)
    gating_output = torch.randn((tokens, experts), dtype=dtype, generator=generator)
    correction_bias = None
    if use_bias:
        correction_bias = torch.linspace(-0.2, 0.2, experts).to(dtype)
    return {
        "hidden_states": hidden_states,
        "gating_output": gating_output,
        "topk": topk,
        "renormalize": renormalize,
        "correction_bias": correction_bias,
    }


def compute_ref(hidden_states, gating_output, topk, renormalize, correction_bias):
    return kernelgenbench.baseline.v1_sqrtsoftplus_expert_topk(
        hidden_states, gating_output, topk, renormalize, correction_bias
    )


def compute_res(hidden_states, gating_output, topk, renormalize, correction_bias):
    return kernelgenbench.solution.v1_sqrtsoftplus_expert_topk(
        hidden_states, gating_output, topk, renormalize, correction_bias
    )


def compare(ref_out, res_out, dtype, use_bias, renormalize, topk):
    assert isinstance(ref_out, tuple) and isinstance(res_out, tuple)
    assert len(ref_out) == len(res_out) == 2
    ref_weights, ref_ids = ref_out
    res_weights, res_ids = res_out
    assert ref_weights.shape == res_weights.shape
    assert ref_ids.shape == res_ids.shape
    assert ref_ids.dtype == res_ids.dtype == torch.int64
    assert ref_weights.dtype == res_weights.dtype
    if use_bias:
        assert ref_weights.dtype == (torch.float32 if renormalize else dtype)
        order_ref = ref_ids.argsort(dim=-1)
        order_res = res_ids.argsort(dim=-1)
        assert torch.equal(
            ref_ids.gather(-1, order_ref), res_ids.gather(-1, order_res)
        )
        tol = 1e-6 if dtype == torch.float32 else 2e-3
        torch.testing.assert_close(
            ref_weights.gather(-1, order_ref),
            res_weights.gather(-1, order_res),
            rtol=tol,
            atol=tol,
        )
    else:
        assert ref_weights.dtype == torch.float32
        assert torch.equal(ref_ids, res_ids)
        torch.testing.assert_close(ref_weights, res_weights, rtol=0, atol=0)
        if ref_weights.numel() and topk > 1:
            assert bool((ref_weights[..., :-1] >= ref_weights[..., 1:]).all())
    if renormalize and res_weights.numel():
        sums = res_weights.float().sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-5, atol=1e-5)


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_sqrtsoftplus_expert_topk")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sqrtsoftplus_expert_topk(config):
    tokens, experts, topk, use_bias, renormalize, dtype = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    before = {name: value.clone() for name, value in res_inputs.items()
              if isinstance(value, torch.Tensor)}
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype, use_bias, renormalize, topk)
    assert ref_out[1].shape == (tokens, topk)
    for name, value in res_inputs.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, before[name]), name
    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return timings(ref_time, res_time)
