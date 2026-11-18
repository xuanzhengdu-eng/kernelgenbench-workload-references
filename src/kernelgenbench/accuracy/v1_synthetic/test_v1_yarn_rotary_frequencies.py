"""Accuracy and performance for v1_yarn_rotary_frequencies.

Operator: Megatron ``YarnRotaryEmbedding.get_emb`` (the frequency table, not the
cos/sin cache), pinned commit
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/models/common/embeddings/yarn_rotary_pos_embedding.py#L106-L158

Contract under test
-------------------
* Inputs are scalars; the only tensor is the returned table.
* Returns ``(emb, mscale)``: ``emb`` is FP32 ``[seq_len, 1, 1, dim]`` whose two
  halves are the same frequencies, and ``mscale`` is a Python ``float``.
* ``emb[s, j] = (s + offset) * inv_freq[j]`` with the YaRN blend between the
  interpolated and the extra frequencies, and the correction range rounded to
  integers only when ``round_range`` is true.
* The concentration factor is ``1.0`` for ``scaling <= 1`` and
  ``0.1*mscale*log(scaling)+1`` otherwise; when either mscale argument is
  ``None`` the second factor is 1.
* The upstream class rejects interleaved rotary embeddings and applies no CP
  slicing at group size one; only that subset is covered.

Item: v1_yarn_rotary_frequencies (723-row 130).
"""
import math

import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rope_rows122_131_checks as checks


# Axes: (seq_len, offset, dim, scaling, round_range, mscale, mscale_all_dim), all
# with base=10000, original_max=4096, beta_fast=32, beta_slow=1.  Covers the empty
# table, a positive and a negative offset, both correction-range modes and both
# concentration-factor paths.  Every case builds the upstream module on the current
# CUDA device: ``get_emb`` migrates its ``inv_freq`` buffers to CUDA on first use, so
# the returned table is always a CUDA tensor (the ``use_cpu_initialization`` flag
# only controls where the buffers are allocated).
COMBOS = [
    (0, 0, 8, 1.0, True, 1.0, 0.0),
    (7, 3, 8, 4.0, True, 1.0, 0.0),
    (11, -2, 64, 16.0, False, 0.5, 0.25),
    (32, 16, 128, 8.0, True, None, None),
    (5, 0, 16, 1.0, True, 2.0, 1.0),
]


def input_build(config):
    """All-scalar argument dictionary; the output table is the only tensor."""
    seq_len, offset, dim, scaling, round_range, mscale, mscale_all_dim = config
    return {
        "seq_len": seq_len,
        "offset": offset,
        "dim": dim,
        "base": 10000.0,
        "scaling": scaling,
        "original_max": 4096,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "mscale": mscale,
        "mscale_all_dim": mscale_all_dim,
        "round_range": round_range,
        "device": str(kernelgenbench.device),
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_yarn_rotary_frequencies(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_yarn_rotary_frequencies(**inputs)


def compare(ref_out, res_out, dtype=None):
    assert isinstance(res_out, tuple) and len(res_out) == 2, \
        f"expected (emb, mscale), got {type(res_out).__name__}"
    ref_emb, ref_factor = ref_out
    res_emb, res_factor = res_out
    assert isinstance(res_factor, float), f"mscale must be a float, got {type(res_factor)}"
    assert isinstance(ref_factor, float)
    assert abs(res_factor - ref_factor) < 1e-12, \
        f"concentration factor {res_factor} != {ref_factor}"
    assert res_emb.shape == ref_emb.shape, \
        f"shape {tuple(res_emb.shape)} != {tuple(ref_emb.shape)}"
    assert res_emb.dtype == ref_emb.dtype == torch.float32
    assert res_emb.device == ref_emb.device
    torch.testing.assert_close(res_emb, ref_emb, rtol=1e-6, atol=1e-6)
    # The second half repeats the first one.
    half = res_emb.shape[-1] // 2
    torch.testing.assert_close(res_emb[..., half:], res_emb[..., :half], rtol=0, atol=0)


def compute_ref_performance(**inputs):
    return checks.measure(kernelgenbench.baseline.v1_yarn_rotary_frequencies, inputs,
                          kernelgenbench.device)


def compute_res_performance(**inputs):
    return checks.measure(kernelgenbench.solution.v1_yarn_rotary_frequencies, inputs,
                          kernelgenbench.device)


@label("v1_yarn_rotary_frequencies")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_yarn_rotary_frequencies(config):
    seq_len, offset, dim, scaling, round_range, mscale, mscale_all_dim = config
    inputs = input_build(config)
    ref_out = compute_ref(**inputs)
    res_out = compute_res(**inputs)
    compare(ref_out, res_out)

    emb, factor = res_out
    assert emb.shape == (seq_len, 1, 1, dim)
    # The table is a linear function of position: consecutive rows differ by exactly
    # one inverse-frequency vector, which also proves the offset is a pure shift.
    if seq_len >= 2:
        rows = emb[:, 0, 0, : dim // 2].cpu()
        step = rows[1] - rows[0]
        expected = rows[0] + torch.arange(seq_len, dtype=torch.float32)[:, None] * step
        torch.testing.assert_close(rows, expected, rtol=1e-6, atol=1e-6)
    log_scale = math.log(scaling) if scaling > 1 else 0.0

    def get_mscale(weight):
        # Upstream: 1.0 for scaling <= 1, otherwise 0.1 * weight * log(scaling) + 1.
        return 1.0 if scaling <= 1 else 0.1 * weight * log_scale + 1.0

    if mscale is None or mscale_all_dim is None:
        expected = get_mscale(1.0)
    else:
        expected = get_mscale(mscale) / get_mscale(mscale_all_dim)
    assert abs(factor - expected) < 1e-12, f"concentration factor {factor} != {expected}"

    # The outlier offset must actually shift the table: an implementation that drops
    # the offset returns a different first row.
    if seq_len and offset:
        zero = dict(inputs, offset=0)
        shifted, _ = compute_ref(**zero)
        assert not torch.allclose(emb, shifted), "offset was ignored"

    return checks.timings(compute_ref_performance(**inputs),
                          compute_res_performance(**inputs))
