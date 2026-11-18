"""Accuracy and performance for v1_sgl_rope_fake_fp4 (fused CUDA bf16 backend).

Operator: SGLang ``_rope_fq4`` on its fused Triton branch (lines 37-44), pinned commit
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py#L37-L44
The fused kernel it dispatches to (``rope_tail_fake_quant_fp4``, lines 117-170):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/attention/dsv4/fp4_rope_fake_quant.py#L117-L170

Contract under test
-------------------
* ``x`` is bf16; its trailing ``rope_dim`` slice is rotated into a fresh output and
  the leading ``x.shape[-1] - rope_dim`` features pass through unchanged.  ``x``
  itself is not modified.
* RoPE is the trailing *interleaved* complex multiplication
  ``(a + ib) * (cos + i sin)`` over pairs of the tail.
* Then FP4 fake quantization: per-32 UE8M0 power-of-two scales for the indexer
  path, per-16 E4M3FN scales clamped to ``[2**-9, 448]`` for ``compressed_kv``,
  round-to-nearest-even on the E2M1 grid, dequantized back to ``x``'s dtype.
* Optional ``positions`` gathers rows of the whole ``freqs`` table.
* Returns a fresh tensor; ``x`` and ``freqs`` are not modified.

Item: v1_sgl_rope_fake_fp4 (723-row 131), fused-backend reference.  The torch
fallback backend of the same entry is the second reference of this item.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rope_rows122_131_checks as checks
from kernelgenbench.dataset.baseline.v1_synthetic.v1_sgl_rope_tail import v1_sgl_rope_tail


# Axes: (shape, rope_dim, compressed_kv, gather).  bf16 inputs so the fused kernel
# is the one actually taken on CUDA.
COMBOS = [
    ((4, 2, 64), 32, False, False),
    ((5, 3, 128), 64, False, True),
    ((3, 2, 64), 16, True, True),
    ((2, 4, 128), 32, True, False),
    ((1, 2, 64), 64, False, True),
]

TOLERANCE = {torch.bfloat16: (2e-2, 1e-2), torch.float32: (1e-6, 1e-6)}


def input_build(config):
    """x: [tokens, heads, dim] bf16; freqs: complex64 [16 or tokens, rope_dim//2]
    (a whole table when ``positions`` is used, one row per token otherwise).
    """
    shape, rope_dim, compressed, gather = config
    generator = torch.Generator(device="cpu").manual_seed(20261231)
    x = torch.randn(shape, dtype=torch.bfloat16, generator=generator)
    rows = 16 if gather else shape[0]
    angle = torch.randn((rows, rope_dim // 2), generator=generator)
    freqs = torch.polar(torch.ones_like(angle), angle)
    positions = torch.arange(shape[0], dtype=torch.int64) * 3 if gather else None
    return {"x": x, "freqs": freqs, "rope_dim": rope_dim,
            "compressed_kv": compressed, "positions": positions}


def compute_ref(x, freqs, rope_dim, compressed_kv=False, positions=None):
    return kernelgenbench.baseline.v1_sgl_rope_fake_fp4(
        x, freqs, rope_dim, compressed_kv=compressed_kv, positions=positions)


def compute_res(x, freqs, rope_dim, compressed_kv=False, positions=None):
    return kernelgenbench.solution.v1_sgl_rope_fake_fp4(
        x, freqs, rope_dim, compressed_kv=compressed_kv, positions=positions)


def compare(ref_out, res_out, dtype=None):
    assert isinstance(res_out, torch.Tensor), f"expected a tensor, got {type(res_out)}"
    assert res_out.shape == ref_out.shape and res_out.dtype == ref_out.dtype
    rtol, atol = TOLERANCE[dtype or ref_out.dtype]
    torch.testing.assert_close(res_out.float(), ref_out.float(), rtol=rtol, atol=atol)


def compute_ref_performance(x, freqs, rope_dim, compressed_kv=False, positions=None):
    return checks.measure(kernelgenbench.baseline.v1_sgl_rope_fake_fp4,
                          {"x": x, "freqs": freqs, "rope_dim": rope_dim,
                           "compressed_kv": compressed_kv, "positions": positions},
                          kernelgenbench.device)


def compute_res_performance(x, freqs, rope_dim, compressed_kv=False, positions=None):
    return checks.measure(kernelgenbench.solution.v1_sgl_rope_fake_fp4,
                          {"x": x, "freqs": freqs, "rope_dim": rope_dim,
                           "compressed_kv": compressed_kv, "positions": positions},
                          kernelgenbench.device)


@label("v1_sgl_rope_fake_fp4")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_rope_fake_fp4(config):
    shape, rope_dim, compressed, gather = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype=torch.bfloat16)

    assert res_out.shape == shape and res_out.dtype == torch.bfloat16
    for name in ("x", "freqs"):
        torch.testing.assert_close(res_inputs[name].cpu(), inputs[name], rtol=0, atol=0)
    assert not torch._C._is_alias_of(res_out, res_inputs["x"]), \
        "the quantized result must be a fresh tensor"

    # The pass-through prefix is untouched by both the rotation and the quantizer.
    # (The quantizer rewrites the whole last axis in blocks, so only check the
    # rotation step on the reference helper.)
    rotated = v1_sgl_rope_tail(res_inputs["x"],
                               res_inputs["freqs"] if res_inputs["positions"] is None
                               else res_inputs["freqs"][res_inputs["positions"]],
                               rope_dim)
    torch.testing.assert_close(rotated[..., :-rope_dim].float(),
                               res_inputs["x"][..., :-rope_dim].float(), rtol=0, atol=0)

    # FP4 rounding really happened: the result is not just the RoPE output.
    assert not torch.equal(res_out, rotated), "the FP4 fake quantization was skipped"

    # Blocks follow the documented granularity: 32 for the indexer, 16 for
    # compressed KV, and the last axis is divisible by it.
    block = 16 if compressed else 32
    assert shape[-1] % block == 0

    return checks.timings(compute_ref_performance(**inputs),
                          compute_res_performance(**inputs))
