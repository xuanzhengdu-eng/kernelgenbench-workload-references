"""Accuracy and performance for v1_ds_vocab_linear_ce_backward (row 158).

Official entry: DeepSpeed ``vocab_parallel_linear_cross_entropy`` backward at
commit 8af59b2e7a91c227081617240e98aa282eadc5f9,
deepspeed/sequence/linear_cross_entropy.py#L168-L212.  The Solution binding
calls the real upstream entry and differentiates it with autograd; this test
also compares against ``F.cross_entropy`` autograd as an independent oracle.
"""
import pytest
import torch
import torch.nn.functional as F

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows_152_161_utils import fresh, measure
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (tokens, hidden, vocab, ignore_index, bias, reduction)
COMBOS = [
    (4, 8, 16, -100, True, "none"),
    (5, 7, 13, 2, True, "mean"),
    (3, 4, 9, -100, False, "sum"),
    (1, 1, 5, -100, True, "none"),
    (6, 16, 32, -100, True, "mean"),
    (2, 8, 11, 0, False, "sum"),
]


def _target(tokens, vocab, ignore_index, seed):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    target = torch.randint(0, vocab, (tokens,), generator=rng, dtype=torch.int64)
    if ignore_index >= 0 and tokens > 1:
        target[1] = ignore_index
    return target


def input_build(config):
    """Build hidden [T,H], weight [V,H], bias [V] and target [T]."""
    tokens, hidden_dim, vocab, ignore_index, bias, reduction = config
    rng = torch.Generator(device="cpu").manual_seed(15800 + tokens * 7 + vocab)
    hidden = torch.randn((tokens, hidden_dim), generator=rng, dtype=torch.float32)
    weight = torch.randn((vocab, hidden_dim), generator=rng, dtype=torch.float32)
    bias_t = torch.randn((vocab,), generator=rng, dtype=torch.float32) if bias else None
    return {
        "hidden": hidden,
        "weight": weight,
        "bias": bias_t,
        "target": _target(tokens, vocab, ignore_index, 15800 + tokens),
        "group": None,
        "vocab_start": 0,
        "grad_loss": torch.tensor(0.7, dtype=torch.float32),
        "ignore_index": ignore_index,
        "reduction": reduction,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_ds_vocab_linear_ce_backward(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_ds_vocab_linear_ce_backward(**inputs)


def _oracle(inputs):
    hidden = inputs["hidden"].detach().clone().requires_grad_()
    weight = inputs["weight"].detach().clone().requires_grad_()
    bias = None
    tensors = [hidden, weight]
    if inputs["bias"] is not None:
        bias = inputs["bias"].detach().clone().requires_grad_()
        tensors.append(bias)
    logits = F.linear(hidden, weight, bias)
    loss = F.cross_entropy(
        logits,
        inputs["target"],
        reduction=inputs["reduction"],
        ignore_index=inputs["ignore_index"],
    )
    if inputs["reduction"] == "none":
        scalar = (loss * inputs["grad_loss"]).sum()
    else:
        scalar = loss * inputs["grad_loss"]
    grads = torch.autograd.grad(scalar, tensors)
    return (grads[0], grads[1], grads[2] if len(grads) > 2 else None)


def compare(ref_out, res_out, oracle):
    assert isinstance(ref_out, tuple) and len(ref_out) == 3
    assert isinstance(res_out, tuple) and len(res_out) == 3
    for name, ref_grad, res_grad, expected in zip(
        ("grad_hidden", "grad_weight", "grad_bias"), ref_out, res_out, oracle
    ):
        if expected is None:
            assert ref_grad is None and res_grad is None, name
            continue
        assert ref_grad.shape == res_grad.shape == expected.shape, name
        assert ref_grad.dtype == res_grad.dtype == torch.float32, name
        torch.testing.assert_close(ref_grad, expected, rtol=2e-5, atol=1e-6, msg=name)
        torch.testing.assert_close(res_grad, expected, rtol=2e-5, atol=1e-6, msg=name)


def compute_ref_performance(inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(inputs):
    return measure(compute_res, inputs)


@label("v1_ds_vocab_linear_ce_backward")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_ds_vocab_linear_ce_backward(config):
    tokens, hidden_dim, vocab, ignore_index, bias, reduction = config
    inputs = input_build(config)
    # The oracle runs on the same device as both implementations so that the
    # comparison is purely numerical.
    oracle = _oracle(
        {
            key: value.to(kernelgenbench.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
    )

    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    for side in (ref_inputs, res_inputs):
        assert side["hidden"].shape == (tokens, hidden_dim)
        assert side["weight"].shape == (vocab, hidden_dim)
        assert side["target"].shape == (tokens,)
        if bias:
            assert side["bias"].shape == (vocab,)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, oracle)

    for name in ("hidden", "weight", "bias", "target"):
        if inputs[name] is not None:
            for side in (ref_inputs, res_inputs):
                assert torch.equal(side[name], inputs[name].to(kernelgenbench.device)), name

    ref_time = compute_ref_performance(ref_inputs)
    res_time = compute_res_performance(res_inputs)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )
