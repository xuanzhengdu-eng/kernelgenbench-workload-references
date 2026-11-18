"""Accuracy and performance for v1_megatron_mla_rope_inplace_.

Operator: Megatron fused MLA YaRN RoPE for the query
(``fused_apply_mla_rope_for_q``, SBHD layout, CP size one).  Official source
(pinned commit):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_mla_yarn_rope_apply.py#L327-L376

Contract under test
-------------------
* ``q`` is ``[S, B, H, qk_head_dim + emb_dim]``, last-dim contiguous.
* In place: the last ``emb_dim`` columns are rewritten, the first
  ``qk_head_dim`` columns are untouched.
* The rotation reads the even/odd pairs ``[x1, x2]`` and writes the split halves
  ``[x1*cos_l - x2*sin_l, x2*cos_r + x1*sin_r]``.  This is the opposite layout
  from the backward (row 123), so the two items are separate contracts.
* Returns the same tensor object that was passed in; ``rotary_interleaved=True``
  is rejected by the pinned entry (fork-only output layout).

Item: v1_megatron_mla_rope_inplace_ (723-row 124).
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rope_rows122_131_checks as checks


# Axes: (S, B, H, qk_head_dim, emb_dim, dtype).
COMBOS = [
    (3, 2, 4, 8, 16, torch.float32),
    (4, 1, 4, 16, 16, torch.float32),
    (2, 2, 8, 8, 16, torch.bfloat16),
    (3, 1, 2, 16, 32, torch.bfloat16),
    (1, 2, 4, 8, 16, torch.float16),
]

TOLERANCE = {
    torch.float32: (2e-6, 1e-6),
    torch.float16: (2e-3, 1e-3),
    torch.bfloat16: (8e-3, 4e-3),
}


def input_build(config):
    """q: [S, B, H, qk_head_dim + emb_dim]; cos/sin: [S, 1, 1, emb_dim]."""
    seq, batch, heads, qk_head_dim, emb_dim, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(20261224)
    q = torch.randn((seq, batch, heads, qk_head_dim + emb_dim), dtype=dtype,
                    generator=generator)
    angle = torch.randn((seq, 1, 1, emb_dim), dtype=torch.float32, generator=generator)
    return {
        "q": q,
        "cos": angle.cos().contiguous(),
        "sin": angle.sin().contiguous(),
        "qk_head_dim": qk_head_dim,
        "emb_dim": emb_dim,
    }


def compute_ref(q, cos, sin, qk_head_dim, emb_dim):
    return kernelgenbench.baseline.v1_megatron_mla_rope_inplace_(
        q, cos, sin, qk_head_dim, emb_dim)


def compute_res(q, cos, sin, qk_head_dim, emb_dim):
    return kernelgenbench.solution.v1_megatron_mla_rope_inplace_(
        q, cos, sin, qk_head_dim, emb_dim)


def compare(ref_out, res_out, dtype=None):
    """Same tensor returned, shape/dtype preserved."""
    assert isinstance(res_out, torch.Tensor), \
        f"expected the in-place tensor back, got {type(res_out).__name__}"
    assert res_out.shape == ref_out.shape and res_out.dtype == ref_out.dtype
    rtol, atol = TOLERANCE[dtype or ref_out.dtype]
    torch.testing.assert_close(res_out.float(), ref_out.float(), rtol=rtol, atol=atol)


def compute_ref_performance(q, cos, sin, qk_head_dim, emb_dim):
    return checks.measure(
        kernelgenbench.baseline.v1_megatron_mla_rope_inplace_,
        {"q": q, "cos": cos, "sin": sin, "qk_head_dim": qk_head_dim, "emb_dim": emb_dim},
        kernelgenbench.device,
    )


def compute_res_performance(q, cos, sin, qk_head_dim, emb_dim):
    return checks.measure(
        kernelgenbench.solution.v1_megatron_mla_rope_inplace_,
        {"q": q, "cos": cos, "sin": sin, "qk_head_dim": qk_head_dim, "emb_dim": emb_dim},
        kernelgenbench.device,
    )


@label("v1_megatron_mla_rope_inplace_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_mla_rope_inplace_(config):
    seq, batch, heads, qk_head_dim, emb_dim, dtype = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    nope_before = res_inputs["q"][..., :qk_head_dim].clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype=dtype)

    for out, side in ((ref_out, ref_inputs), (res_out, res_inputs)):
        assert out.shape == (seq, batch, heads, qk_head_dim + emb_dim)
        # Same object back, and the nope prefix is bit-identical.
        assert out.data_ptr() == side["q"].data_ptr(), \
            "the forward must modify and return the input tensor"
        torch.testing.assert_close(out[..., :qk_head_dim], nope_before, rtol=0, atol=0)
        # The tail is exactly the concatenation of the two rotated halves.
        half = emb_dim // 2
        left, right = out[..., qk_head_dim:qk_head_dim + half], out[..., qk_head_dim + half:]
        rebuilt = torch.cat((left, right), -1)
        torch.testing.assert_close(rebuilt, out[..., qk_head_dim:], rtol=0, atol=0)

    # The reference rejects the fork-only interleaved output layout, and so does the
    # candidate, because the pinned kernel asserts it.
    for call in (kernelgenbench.baseline.v1_megatron_mla_rope_inplace_,
                 kernelgenbench.solution.v1_megatron_mla_rope_inplace_):
        probe = checks.fresh(inputs, kernelgenbench.device)
        with pytest.raises(AssertionError):
            call(rotary_interleaved=True, **probe)
        # ... and the interleaved flag is the only difference; the split layout passes.
        call(**probe)

    # cos/sin are read-only.
    for name in ("cos", "sin"):
        torch.testing.assert_close(res_inputs[name].cpu(), inputs[name], rtol=0, atol=0)

    return checks.timings(compute_ref_performance(**inputs),
                          compute_res_performance(**inputs))
