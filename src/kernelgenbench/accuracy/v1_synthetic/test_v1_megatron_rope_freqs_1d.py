"""Accuracy and performance for v1_megatron_rope_freqs_1d.

Operator: Megatron ``RotaryEmbedding.get_emb`` plus the CP slice in
``RotaryEmbedding.forward`` (native torch).
Official source (pinned commit 266f1c97dca477ca8b92c16087506da2000b0b84):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/models/common/embeddings/rotary_pos_embedding.py#L150-L206
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/models/common/embeddings/rope_utils.py#L48-L70

Contract: inv_freq is a float32 buffer of ``half_dim`` frequencies on the input
device.  ``seq = arange(max_seq_len) + offset``, optionally divided by
``interpolation``; ``freq = outer(seq, inv_freq)``.  The output is
``(max_seq_len, 1, 1, 2 * half_dim)`` with the two frequency halves equal, or
interleaved pairs when ``interleaved``.  With ``cp_size > 1`` and
``packed_seq == False`` the sequence axis is sliced to ``max_seq_len / cp_size``
by selecting rows ``[cp_rank, 2*cp_size - cp_rank - 1]`` of the
``(2*cp_size, max_seq_len/(2*cp_size))`` view.  inv_freq is not modified.

Item: v1_megatron_rope_freqs_1d (723-row 142).  ``compute_ref`` loads the
reference from ``kernelgenbench.baseline``; ``compute_res`` calls whatever the
verifier bound as ``kernelgenbench.solution`` for the same label, i.e. the real
pinned Megatron operator during verification.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows142_151_checks as checks


# (seq_len, half_dim, offset, interpolation, interleaved, packed_seq, cp_size, cp_rank)
# cp_size=2 cases keep seq_len divisible by 2*cp_size; packed_seq=True bypasses CP.
COMBOS = [
    (16, 8, 0, None, False, False, 1, 0),
    (16, 8, 3, None, True, False, 1, 0),
    (24, 16, 5, 2.0, False, False, 1, 0),
    (24, 16, 7, 2.0, True, False, 1, 0),
    (16, 8, 0, None, False, False, 2, 0),
    (16, 8, 3, 2.0, True, False, 2, 1),
    (24, 16, 5, None, True, True, 2, 0),
    (24, 16, 7, 2.0, False, True, 2, 1),
    (8, 2, 1, None, True, False, 2, 1),
    (1, 4, 0, None, False, False, 1, 0),
]


def input_build(config):
    """inv_freq: (half_dim,) float32; derived from the same formula as Megatron."""
    seq_len, half_dim, offset, interpolation, interleaved, packed, cp_size, cp_rank = config
    generator = torch.Generator(device="cpu").manual_seed(20261115)
    positions = torch.arange(half_dim, dtype=torch.float32)
    inv_freq = 1.0 / (10000.0 ** (positions / half_dim))
    return {
        "inv_freq": inv_freq,
        "max_seq_len": seq_len,
        "offset": offset,
        "interpolation": interpolation,
        "interleaved": interleaved,
        "packed_seq": packed,
        "cp_size": cp_size,
        "cp_rank": cp_rank,
    }


def compute_ref(inv_freq, max_seq_len, offset=0, interpolation=None, interleaved=False,
                packed_seq=False, cp_size=1, cp_rank=0):
    return kernelgenbench.baseline.v1_megatron_rope_freqs_1d(
        inv_freq, max_seq_len, offset, interpolation, interleaved, packed_seq, cp_size, cp_rank)


def compute_res(inv_freq, max_seq_len, offset=0, interpolation=None, interleaved=False,
                packed_seq=False, cp_size=1, cp_rank=0):
    return kernelgenbench.solution.v1_megatron_rope_freqs_1d(
        inv_freq, max_seq_len, offset, interpolation, interleaved, packed_seq, cp_size, cp_rank)


def compare(ref_out, res_out, config):
    seq_len, half_dim, _offset, _interp, interleaved, packed, cp_size, _rank = config
    assert isinstance(res_out, torch.Tensor), f"expected a tensor, got {type(res_out).__name__}"
    rows = seq_len // cp_size if cp_size > 1 and not packed else seq_len
    assert res_out.shape == (rows, 1, 1, 2 * half_dim), res_out.shape
    assert res_out.dtype == torch.float32
    # The two frequency halves must be identical, or pairwise equal when interleaved.
    if interleaved:
        assert torch.equal(res_out[..., 0::2], res_out[..., 1::2]), \
            "interleaved layout must repeat each frequency in adjacent lanes"
    else:
        assert torch.equal(res_out[..., :half_dim], res_out[..., half_dim:]), \
            "non-interleaved layout must duplicate the frequency half"
    torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0)
    assert torch.isfinite(res_out).all()


def compute_ref_performance(**kwargs):
    return checks.measure(compute_ref, kwargs, kernelgenbench.device)


def compute_res_performance(**kwargs):
    return checks.measure(compute_res, kwargs, kernelgenbench.device)


@label("v1_megatron_rope_freqs_1d")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_rope_freqs_1d(config):
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    original = inputs["inv_freq"].clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config)

    # The frequency buffer is read-only; neither side may modify or alias it.
    for side in (ref_inputs, res_inputs):
        torch.testing.assert_close(side["inv_freq"].cpu(), original, rtol=0, atol=0)
        assert not torch._C._is_alias_of(res_out, side["inv_freq"])

    return checks.timings(compute_ref_performance(**inputs),
                          compute_res_performance(**inputs))
