"""Accuracy and performance for v1_megatron_mla_rope_kv_split.

Operator: Megatron fused MLA YaRN RoPE for K/V (``fused_apply_mla_rope_for_kv``,
SBHD layout, CP size one).  Official source (pinned commit):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_mla_yarn_rope_apply.py#L735-L782

Contract under test
-------------------
* ``kv`` is ``[S, B, H, k_dim + v_dim]``; the first ``k_dim`` columns are the key
  pass-through, the last ``v_dim`` columns are the value.
* ``k_pos_emb`` is ``[S, B, 1, emb_dim]``, a *separate* tensor, not a slice of
  ``kv``.  Its even/odd pairs are rotated into split halves
  ``[x1*cos_l - x2*sin_l, x2*cos_r + x1*sin_r]`` and appended after the key.
* Outputs: ``key [S, B, H, k_dim + emb_dim]`` and ``value [S, B, H, v_dim]``,
  both newly allocated and contiguous; ``value`` is a copy, not a view of ``kv``.
* Inputs are not modified.  Only ``rotary_interleaved=False`` is accepted (the
  pinned Triton entry asserts it); the fork-only interleaved output branch has no
  official counterpart and is not part of this item.

Item: v1_megatron_mla_rope_kv_split (723-row 122).
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rope_rows122_131_checks as checks


# Axes: (S, B, H, k_dim, v_dim, emb_dim, dtype).  k_dim, v_dim and emb_dim // 2
# must be powers of two because the Triton kernel indexes them with tl.arange.
COMBOS = [
    (3, 2, 4, 8, 8, 16, torch.float32),
    (2, 1, 2, 16, 8, 16, torch.float32),
    (3, 1, 4, 8, 16, 16, torch.bfloat16),
    (2, 2, 8, 16, 16, 16, torch.bfloat16),
    (1, 2, 4, 8, 8, 32, torch.bfloat16),
    (4, 2, 2, 32, 32, 64, torch.float16),
]

# One storage rounding for fp16/bf16; fp32 should be exact because both sides do
# the same float32 multiply-add and differ only in vector width.
TOLERANCE = {
    torch.float32: (2e-6, 1e-6),
    torch.float16: (2e-3, 1e-3),
    torch.bfloat16: (8e-3, 4e-3),
}


def input_build(config):
    """kv: [S, B, H, k_dim+v_dim]; k_pos_emb: [S, B, 1, emb_dim];
    cos/sin: [S, 1, 1, emb_dim] (the upstream tables are per sequence position).
    """
    seq, batch, heads, k_dim, v_dim, emb_dim, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(20261222)
    kv = torch.randn((seq, batch, heads, k_dim + v_dim), dtype=dtype, generator=generator)
    k_pos_emb = torch.randn((seq, batch, 1, emb_dim), dtype=dtype, generator=generator)
    angle = torch.randn((seq, 1, 1, emb_dim), dtype=torch.float32, generator=generator)
    return {
        "kv": kv,
        "k_pos_emb": k_pos_emb,
        "cos": angle.cos().contiguous(),
        "sin": angle.sin().contiguous(),
        "emb_dim": emb_dim,
        "k_dim": k_dim,
        "v_dim": v_dim,
    }


def compute_ref(kv, k_pos_emb, cos, sin, emb_dim, k_dim, v_dim):
    return kernelgenbench.baseline.v1_megatron_mla_rope_kv_split(
        kv, k_pos_emb, cos, sin, emb_dim, k_dim, v_dim)


def compute_res(kv, k_pos_emb, cos, sin, emb_dim, k_dim, v_dim):
    return kernelgenbench.solution.v1_megatron_mla_rope_kv_split(
        kv, k_pos_emb, cos, sin, emb_dim, k_dim, v_dim)


def compare(ref_out, res_out, dtype=None):
    """Two fresh tensors; the key keeps the nope prefix bit-exactly and the value
    is the untouched tail of ``kv``.
    """
    assert isinstance(res_out, tuple) and len(res_out) == 2, \
        f"expected a (key, value) pair, got {type(res_out).__name__}"
    key_ref, value_ref = ref_out
    key_res, value_res = res_out
    assert key_res.shape == key_ref.shape, \
        f"key shape {tuple(key_res.shape)} != {tuple(key_ref.shape)}"
    assert value_res.shape == value_ref.shape, \
        f"value shape {tuple(value_res.shape)} != {tuple(value_ref.shape)}"
    assert key_res.dtype == key_ref.dtype and value_res.dtype == value_ref.dtype
    assert key_res.is_contiguous() and value_res.is_contiguous(), \
        "both outputs must be contiguous"
    rtol, atol = TOLERANCE[dtype or key_ref.dtype]
    torch.testing.assert_close(key_res.float(), key_ref.float(), rtol=rtol, atol=atol)
    torch.testing.assert_close(value_res.float(), value_ref.float(), rtol=0, atol=0)


def compute_ref_performance(kv, k_pos_emb, cos, sin, emb_dim, k_dim, v_dim):
    return checks.measure(
        kernelgenbench.baseline.v1_megatron_mla_rope_kv_split,
        {"kv": kv, "k_pos_emb": k_pos_emb, "cos": cos, "sin": sin,
         "emb_dim": emb_dim, "k_dim": k_dim, "v_dim": v_dim},
        kernelgenbench.device,
    )


def compute_res_performance(kv, k_pos_emb, cos, sin, emb_dim, k_dim, v_dim):
    return checks.measure(
        kernelgenbench.solution.v1_megatron_mla_rope_kv_split,
        {"kv": kv, "k_pos_emb": k_pos_emb, "cos": cos, "sin": sin,
         "emb_dim": emb_dim, "k_dim": k_dim, "v_dim": v_dim},
        kernelgenbench.device,
    )


@label("v1_megatron_mla_rope_kv_split")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_mla_rope_kv_split(config):
    seq, batch, heads, k_dim, v_dim, emb_dim, dtype = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    for side in (ref_inputs, res_inputs):
        assert side["kv"].shape == (seq, batch, heads, k_dim + v_dim)
        assert side["k_pos_emb"].shape == (seq, batch, 1, emb_dim)
        assert side["cos"].shape == side["sin"].shape == (seq, 1, 1, emb_dim)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype=dtype)

    for output, side, source in ((ref_out, ref_inputs, inputs), (res_out, res_inputs, inputs)):
        key, value = output
        assert key.shape == (seq, batch, heads, k_dim + emb_dim)
        assert value.shape == (seq, batch, heads, v_dim)
        # The key prefix is the untouched nope part of kv.
        torch.testing.assert_close(key[..., :k_dim], side["kv"][..., :k_dim], rtol=0, atol=0)
        # value is a copy of the tail, not a view, and no output aliases an input.
        torch.testing.assert_close(value, side["kv"][..., k_dim:], rtol=0, atol=0)
        for produced in (key, value):
            for name, tensor in side.items():
                if isinstance(tensor, torch.Tensor):
                    assert not torch._C._is_alias_of(produced, tensor), f"output aliases {name}"
        # Inputs must be left untouched.
        for name, tensor in side.items():
            if isinstance(tensor, torch.Tensor):
                torch.testing.assert_close(tensor.cpu(), source[name], rtol=0, atol=0)

    return checks.timings(compute_ref_performance(**inputs),
                          compute_res_performance(**inputs))
