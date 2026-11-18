"""Accuracy and performance for v1_megatron_language_embedding (723-row 469).

Operator: Megatron-LM ``LanguageModelEmbedding.forward`` (single TP rank, no SP scatter).
Official source (pinned commit 266f1c97dca477ca8b92c16087506da2000b0b84):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/models/common/embeddings/language_model_embedding.py#L99-L154

Contract: returns [S, B, H].  Position embeddings are added in [B, S, H] before
the transpose; token-type embeddings are added after it; ``mup_scale`` multiplies
when it differs from 1.0; ``fp32_residual`` casts to fp32; dropout is inverted
(scales by 1/(1-p) in training, identity in eval).  The test asserts the layout,
dtype, the exact equality of the deterministic path, and the dropout mask
structure (each element is either 0 or ``value/(1-p)``).

Dtype contract (measured on the pinned module, see the reference docstring):
upstream builds the position and token-type tables as plain fp32
``nn.Embedding`` objects, so any config that uses them returns fp32 even when
``params_dtype`` is fp16/bf16.  ``expected_dtype`` below encodes that rule and
the fp16/bf16 COMBOS without those tables keep the input dtype.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows462_471_utils import fresh, measure
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (B, S, H, dtype, position embedding, token type, mup scale, fp32 output,
#  dropout p, training)
COMBOS = [
    (2, 4, 8, torch.float32, True, True, 1.0, False, 0.0, False),
    (1, 5, 7, torch.bfloat16, False, False, 1.0, True, 0.0, False),
    (3, 2, 16, torch.bfloat16, True, False, 1.3, False, 0.0, False),
    (1, 1, 8, torch.float32, False, True, 0.5, True, 0.0, False),
    (4, 32, 64, torch.float32, True, True, 1.0, False, 0.0, False),
    (4, 32, 64, torch.bfloat16, True, False, 1.3, True, 0.0, False),
    (4, 32, 64, torch.float32, False, False, 0.5, False, 0.0, False),
    (4, 32, 64, torch.bfloat16, False, True, 1.0, True, 0.0, False),
    # Half precision without a position/token-type table: the word embedding is
    # the only table, so the output must stay in the input dtype (the fp32-table
    # promotion above needs one of the fp32 tables to participate).
    (2, 8, 12, torch.float16, False, False, 1.0, False, 0.0, False),
    (2, 8, 12, torch.bfloat16, False, False, 1.0, False, 0.0, False),
    # Dropout in training mode: structure-only assertions (see compare).
    (8, 64, 32, torch.float32, True, True, 1.0, False, 0.25, True),
    (8, 64, 32, torch.float32, False, False, 1.3, False, 0.5, True),
]

VOCAB = 13
POSITIONS = 9
TOKENTYPES = 3

TOLERANCE = {
    torch.float32: (4e-6, 2e-6),
    torch.float16: (2e-3, 1e-3),
    torch.bfloat16: (1e-2, 8e-3),
}


def expected_dtype(config):
    """Upstream returns fp32 when a fp32 position/token-type table participates."""
    batch, seq, hidden, dtype, use_position, use_tokentype, scale, fp32, p, training = config
    if fp32 or use_position or use_tokentype or dtype == torch.float32:
        return torch.float32
    return dtype


def input_build(config):
    """ids [B,S] and the three weight tables on CPU with a fixed seed."""
    batch, seq, hidden, dtype, use_position, use_tokentype, scale, fp32, p, training = config
    generator = torch.Generator(device="cpu").manual_seed(20261115)

    def weight(shape):
        return torch.randn(shape, dtype=dtype, generator=generator)

    inputs = {
        "input_ids": torch.randint(VOCAB, (batch, seq), generator=generator),
        "position_ids": torch.randint(POSITIONS, (batch, seq), generator=generator),
        "tokentype_ids": torch.randint(TOKENTYPES, (batch, seq), generator=generator)
        if use_tokentype else None,
        "word_weight": weight((VOCAB, hidden)),
        "position_weight": weight((POSITIONS, hidden)) if use_position else None,
        "tokentype_weight": weight((TOKENTYPES, hidden)) if use_tokentype else None,
        "mup_scale": scale,
        "fp32_residual": fp32,
        "dropout_prob": p,
        "training": training,
    }
    return inputs


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_megatron_language_embedding(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_megatron_language_embedding(**inputs)


def deterministic_output(**inputs):
    """The same call with dropout disabled, used as the dropout mask baseline."""
    stable = dict(inputs)
    stable["dropout_prob"] = 0.0
    stable["training"] = False
    return compute_ref(**stable)


def compare(ref_out, res_out, config, inputs):
    batch, seq, hidden, dtype, *_rest, p, training = config
    want = expected_dtype(config)
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape == (seq, batch, hidden)
    assert res_out.dtype == ref_out.dtype == want
    if not training or p == 0.0:
        rtol, atol = TOLERANCE[want]
        torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)
        return

    # Dropout: compare both sides to their own deterministic baseline.  Every
    # kept element equals value/(1-p) and every dropped element is exactly 0.
    scale = 1.0 / (1.0 - p)
    for output in (ref_out, res_out):
        baseline = deterministic_output(**inputs)
        kept = baseline * scale
        zeros = output == 0
        torch.testing.assert_close(output[~zeros], kept[~zeros], rtol=1e-5, atol=1e-5)
        fraction = float(zeros.float().mean())
        assert abs(fraction - p) < 0.12, f"drop fraction {fraction} far from {p}"
    # Two independent calls must differ (dropout is stochastic, not a no-op).
    assert not torch.equal(ref_out, res_out) or not torch.equal(
        res_out, deterministic_output(**inputs) * scale)


@label("v1_megatron_language_embedding")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_language_embedding(config):
    batch, seq, hidden, dtype, *_ = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)

    # Stable path first, then the configured (possibly dropout) call.
    stable = dict(res_inputs)
    stable["training"] = False
    stable["dropout_prob"] = 0.0
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config, res_inputs)

    assert ref_out.dtype == expected_dtype(config)
    assert not torch._C._is_alias_of(res_out, res_inputs["word_weight"])

    for name, initial in inputs.items():
        if isinstance(initial, torch.Tensor):
            torch.testing.assert_close(res_inputs[name].cpu(), initial, rtol=0, atol=0)

    ref_time = measure(compute_ref, stable)
    res_time = measure(compute_res, stable)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
