"""Accuracy and performance for v1_ds_vocab_linear_ce (row 159).

Official entry: DeepSpeed ``vocab_parallel_linear_cross_entropy`` forward at
commit 8af59b2e7a91c227081617240e98aa282eadc5f9,
deepspeed/sequence/linear_cross_entropy.py#L215-L326.  The test compares the
reference and the real upstream operator against ``F.cross_entropy`` for all
three reduction modes, including ignored targets.
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
    (6, 8, 13, 2, True, "none"),
]


def input_build(config):
    """Build hidden [T,H], weight [V,H], bias [V] and target [T]."""
    tokens, hidden_dim, vocab, ignore_index, bias, reduction = config
    rng = torch.Generator(device="cpu").manual_seed(15900 + tokens * 5 + vocab)
    hidden = torch.randn((tokens, hidden_dim), generator=rng, dtype=torch.float32)
    weight = torch.randn((vocab, hidden_dim), generator=rng, dtype=torch.float32)
    bias_t = torch.randn((vocab,), generator=rng, dtype=torch.float32) if bias else None
    target = torch.randint(0, vocab, (tokens,), generator=rng, dtype=torch.int64)
    if ignore_index >= 0 and tokens > 1:
        target[1] = ignore_index
    return {
        "hidden": hidden,
        "weight": weight,
        "bias": bias_t,
        "target": target,
        "group": None,
        "vocab_start": 0,
        "ignore_index": ignore_index,
        "reduction": reduction,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_ds_vocab_linear_ce(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_ds_vocab_linear_ce(**inputs)


def _oracle(inputs):
    logits = F.linear(inputs["hidden"], inputs["weight"], inputs["bias"])
    return F.cross_entropy(
        logits,
        inputs["target"],
        reduction=inputs["reduction"],
        ignore_index=inputs["ignore_index"],
    )


def compare(ref_out, res_out, oracle):
    if oracle.shape == ():
        assert ref_out.shape == res_out.shape == ()
    else:
        assert ref_out.shape == res_out.shape == oracle.shape
    assert ref_out.dtype == res_out.dtype == torch.float32
    torch.testing.assert_close(ref_out, oracle, rtol=2e-5, atol=1e-6)
    torch.testing.assert_close(res_out, oracle, rtol=2e-5, atol=1e-6)


def compute_ref_performance(inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(inputs):
    return measure(compute_res, inputs)


@label("v1_ds_vocab_linear_ce")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_ds_vocab_linear_ce(config):
    tokens, hidden_dim, vocab, ignore_index, bias, reduction = config
    inputs = input_build(config)
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

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, oracle)

    if reduction == "none":
        ignored = ref_inputs["target"] == ignore_index
        if ignored.any():
            assert torch.equal(ref_out[ignored], torch.zeros_like(ref_out[ignored]))
            assert torch.equal(res_out[ignored], torch.zeros_like(res_out[ignored]))

    for name in ("hidden", "weight", "bias", "target"):
        if inputs[name] is not None:
            for side in (ref_inputs, res_inputs):
                assert torch.equal(side[name], inputs[name].to(kernelgenbench.device)), name

    ref_time = compute_ref_performance(ref_inputs)
    res_time = compute_res_performance(res_inputs)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )
