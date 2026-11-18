"""Accuracy and performance for Megatron unfused Switch MoE auxiliary loss.

Official implementation:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/moe_utils.py#L59-L146

``probs`` is ``(tokens, experts)``, ``tokens_per_expert`` is ``(experts,)``,
and ``total_num_tokens`` may be the global count.  The optional mask is
``(tokens,)`` and, unlike z-loss, ``True`` means a valid token.  An
all-invalid mask must compute the masked sum without dividing by zero.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.bias_activation_test_utils import fresh, measure
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


COMBOS = [
    ((2, 16, 128), dtype, masked)
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
    for masked in (False, True, "none_valid")
] + [
    ((2, 128, 1024), dtype, False)
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
]


def input_build(config):
    shape, dtype, masked = config
    tokens, experts = shape[0] * shape[1], shape[-1]
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    probs = torch.softmax(torch.randn((tokens, experts), generator=generator).float(), dim=-1).to(dtype)
    counts = torch.randint(0, 8, (experts,), generator=generator)
    if masked is False:
        padding_mask = None
    elif masked == "none_valid":
        padding_mask = torch.zeros((tokens,), dtype=torch.bool)
    else:
        padding_mask = (torch.arange(tokens) % 5 == 0)
    return {
        "probs": probs,
        "tokens_per_expert": counts,
        "total_num_tokens": tokens,
        "topk": 2,
        "num_experts": experts,
        "moe_aux_loss_coeff": 0.01,
        "padding_mask": padding_mask,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_moe_switch_aux_loss(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_moe_switch_aux_loss(**inputs)


def compare(ref_out, res_out, dtype):
    assert isinstance(ref_out, torch.Tensor) and isinstance(res_out, torch.Tensor)
    assert ref_out.shape == res_out.shape == ()
    assert ref_out.dtype == res_out.dtype
    output_dtype = ref_out.dtype if ref_out.dtype in (
        torch.float32, torch.float16, torch.bfloat16) else dtype
    rtol, atol = {
        torch.float32: (4e-6, 2e-6),
        torch.float16: (2e-3, 1e-3),
        torch.bfloat16: (1e-2, 8e-3),
    }[output_dtype]
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)


def check_gradients(inputs, dtype):
    """dL/dp[t, i] = mask[t] * counts[i] * E*coeff/(topk*T^2)."""
    if dtype != torch.float32:
        return
    device = kernelgenbench.device
    probs = inputs["probs"].to(device).clone().requires_grad_(True)
    counts = inputs["tokens_per_expert"].to(device)
    mask = inputs["padding_mask"]
    mask = None if mask is None else mask.to(device)
    value = compute_res(probs=probs, tokens_per_expert=counts,
                        total_num_tokens=inputs["total_num_tokens"], topk=inputs["topk"],
                        num_experts=inputs["num_experts"],
                        moe_aux_loss_coeff=inputs["moe_aux_loss_coeff"],
                        padding_mask=mask)
    (grad,) = torch.autograd.grad(value, probs)

    scale = (inputs["num_experts"] * inputs["moe_aux_loss_coeff"]
             / (inputs["topk"] * inputs["total_num_tokens"] ** 2))
    weight = counts.float()
    expected = weight.unsqueeze(0) * scale
    if mask is not None:
        expected = expected * mask.float().unsqueeze(-1)
    torch.testing.assert_close(grad, expected.expand_as(grad), rtol=1e-4, atol=1e-9)
    assert grad.shape == probs.shape and torch.isfinite(grad).all()


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_moe_switch_aux_loss")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_moe_switch_aux_loss(config):
    shape, dtype, masked = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    for side in (ref_inputs, res_inputs):
        assert side["probs"].shape == (shape[0] * shape[1], shape[-1])
        assert side["tokens_per_expert"].shape == (shape[-1],)
        if masked is False:
            assert side["padding_mask"] is None
        else:
            assert side["padding_mask"].shape == (shape[0] * shape[1],)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    for side in (ref_inputs, res_inputs):
        assert torch.equal(side["probs"].cpu(), inputs["probs"])
        assert torch.equal(side["tokens_per_expert"],
                           inputs["tokens_per_expert"].to(kernelgenbench.device))
        if inputs["padding_mask"] is not None:
            assert torch.equal(side["padding_mask"].cpu(), inputs["padding_mask"])

    check_gradients(inputs, dtype)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
