"""Accuracy and performance for v1_sgl_logits_head_gate (SGLang DSA head gate).

x is BF16 (tokens, hidden), weight is BF16 (heads, hidden); the official
logits_head_gate_graph projects to FP32 and then multiplies by
1/sqrt(heads), q_scale and softmax_scale.  q_scale may broadcast one head
group across heads.
"""
import pytest
import torch
import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.bias_activation_test_utils import fresh, measure
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (tokens, hidden, heads, q_scale groups)
COMBOS = [
    (1, 16, 1, 1),
    (3, 32, 4, 2),
    (2, 64, 8, 4),
    (4, 128, 3, 1),
]


def input_build(config):
    tokens, hidden, heads, groups = config
    g = torch.Generator(device="cpu").manual_seed(251000 + hidden + heads)
    return {
        "x": torch.randn((tokens, hidden), dtype=torch.bfloat16, generator=g),
        "weight": torch.randn((heads, hidden), dtype=torch.bfloat16, generator=g),
        "n_heads_inv_sqrt": heads ** -0.5,
        "softmax_scale": 0.125,
        "q_scale": torch.rand((tokens, 1 if groups == 1 else heads, groups),
                              dtype=torch.float32, generator=g),
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_sgl_logits_head_gate(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_sgl_logits_head_gate(**inputs)


def compare(ref_out, res_out, config):
    tokens, _, heads, groups = config
    assert res_out.shape == ref_out.shape == (tokens, heads, groups)
    assert res_out.dtype == ref_out.dtype == torch.float32
    torch.testing.assert_close(res_out, ref_out, rtol=1e-5, atol=1e-5)


def compute_ref_performance(inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(inputs):
    return measure(compute_res, inputs)


@label("v1_sgl_logits_head_gate")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_logits_head_gate(config):
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config)
    for side in (ref_inputs, res_inputs):
        for name in ("x", "weight", "q_scale"):
            torch.testing.assert_close(side[name].cpu(), inputs[name], rtol=0, atol=0)
    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
