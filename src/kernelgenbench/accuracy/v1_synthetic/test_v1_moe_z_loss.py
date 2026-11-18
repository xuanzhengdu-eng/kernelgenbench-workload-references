"""Accuracy and performance for Megatron unfused MoE router z-loss.

Official implementation:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/moe_utils.py#L149-L178

``logits`` is ``(tokens, experts)``; the optional mask is ``(tokens,)`` and
``True`` means padding (excluded).  Without a mask the loss is the mean of
``logsumexp(logits)**2`` times the coefficient.  An all-padding mask returns
zero rather than dividing by zero.
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
    for masked in (False, True, "all")
] + [
    ((2, 128, 1024), dtype, False)
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
]


def input_build(config):
    shape, dtype, masked = config
    tokens = shape[0] * shape[1]
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    logits = torch.randn((tokens, shape[-1]), dtype=dtype, generator=generator)
    if masked is False:
        padding_mask = None
    elif masked == "all":
        padding_mask = torch.ones((tokens,), dtype=torch.bool)
    else:
        padding_mask = (torch.arange(tokens) % 5 == 0)
    return {"logits": logits, "z_loss_coeff": 0.1, "padding_mask": padding_mask}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_moe_z_loss(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_moe_z_loss(**inputs)


def compare(ref_out, res_out, dtype):
    assert isinstance(ref_out, torch.Tensor) and isinstance(res_out, torch.Tensor)
    assert ref_out.shape == res_out.shape == ()
    # A masked reduction divides by an int64 count, which promotes the scalar to
    # FP32 even when the logits are FP16/BF16; both sides must agree on that.
    assert ref_out.dtype == res_out.dtype
    rtol, atol = {
        torch.float32: (4e-6, 2e-6),
        torch.float16: (2e-3, 1e-3),
        torch.bfloat16: (1e-2, 8e-3),
    }[ref_out.dtype]
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)


def check_gradients(inputs, dtype, masked):
    """d/dz of the z-loss matches 2*coeff*logsumexp(z)*softmax(z) over valid rows."""
    if dtype != torch.float32:
        return
    logits = inputs["logits"].to(kernelgenbench.device).clone().requires_grad_(True)
    mask = inputs["padding_mask"]
    mask = None if mask is None else mask.to(kernelgenbench.device)
    value = compute_ref(logits=logits, z_loss_coeff=inputs["z_loss_coeff"], padding_mask=mask)
    (grad,) = torch.autograd.grad(value, logits)

    lse = torch.logsumexp(logits.detach(), dim=-1)
    probs = torch.softmax(logits.detach(), dim=-1)
    coeff = inputs["z_loss_coeff"]
    if mask is None:
        expected = (2.0 * coeff / logits.shape[0]) * lse.unsqueeze(-1) * probs
    else:
        valid = (~mask).float()
        expected = (2.0 * coeff / valid.sum().clamp(min=1.0)) * valid.unsqueeze(-1) \
            * lse.unsqueeze(-1) * probs
    torch.testing.assert_close(grad, expected, rtol=1e-4, atol=1e-6)
    assert grad.shape == logits.shape and torch.isfinite(grad).all()


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_moe_z_loss")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_moe_z_loss(config):
    shape, dtype, masked = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    for side in (ref_inputs, res_inputs):
        assert side["logits"].shape == (shape[0] * shape[1], shape[-1])
        if masked is False:
            assert side["padding_mask"] is None
        else:
            assert side["padding_mask"].shape == (shape[0] * shape[1],)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    for side in (ref_inputs, res_inputs):
        assert torch.equal(side["logits"].cpu(), inputs["logits"])
        if inputs["padding_mask"] is not None:
            assert torch.equal(side["padding_mask"].cpu(), inputs["padding_mask"])

    check_gradients(inputs, dtype, masked)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
