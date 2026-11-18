"""Accuracy and performance for v1_megatron_mla_rope_kv_split_backward.

Official sources (NVIDIA/Megatron-LM pinned commit 266f1c97):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_mla_yarn_rope_apply.py#L675-L732
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_mla_yarn_rope_apply.py#L487-L578

Contract (backward of ``ApplyMLARotaryEmbKV``, SBHD, ``cu_seqlens_kv=None``,
``cp_rank=0``/``cp_size=1``): ``dk`` is the upstream gradient of ``o_key``
``[S, B, H, emb_dim + k_dim]`` and ``dv`` that of ``o_value``
``[S, B, H, v_dim]``; ``cos``/``sin`` are ``[S, 1, 1, emb_dim]`` contiguous.
The gradient wrt the fused ``kv`` input is ``cat(dk[..., :k_dim], dv)``.  The
gradient wrt the positional embedding sums the trailing ``emb_dim`` lanes over
heads in FP32, rounds both halves to the dEMB dtype, rotates them with
``(l*cos + r*sin, r*cos - l*sin)`` using the two cos/sin halves, and stores the
two results interleaved (even lanes first expression, odd lanes second), giving
``[S, B, 1, emb_dim]``.  The backward returns one value per forward argument
(11): the two gradients plus nine ``None``.  Only ``rotary_interleaved=False``
is supported upstream, so the old reference's interleaved branch is removed.
No input is modified and neither gradient aliases an input.

Item: v1_megatron_mla_rope_kv_split_backward (723-row 121).  The old reference
cited the flagos-ai fork; the official pinned Megatron contains the same
autograd function, so the reference is re-pointed at the official file.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows112_121_checks as checks


# (S, B, H, k_dim, v_dim, emb_dim, dtype): dk [S,B,H,k+emb], dv [S,B,H,v],
# cos/sin [S,1,1,emb].  The Triton kernel builds tl.arange over k_dim/v_dim/emb_dim,
# so all three must be powers of two (upstream precondition).
CASES = [
    (3, 2, 4, 8, 8, 16, torch.float32),
    (3, 1, 4, 8, 16, 16, torch.bfloat16),
    (2, 2, 8, 16, 16, 16, torch.bfloat16),
    (1, 1, 1, 8, 8, 8, torch.float32),
    (4, 2, 2, 4, 8, 8, torch.float32),
]

# Axes: one combo per case.
COMBOS = list(CASES)

# The kernel reduces over heads in fp32 and rounds the two halves to the dEMB
# dtype before rotating; with bf16 that is one storage rounding of a sum whose
# order differs from torch.sum.
TOLERANCE = {
    torch.float32: (4e-6, 4e-6),
    torch.bfloat16: (2e-2, 1e-2),
}

GRADIENT_COUNT = 11


def input_build(config):
    """dk [S,B,H,k+emb]; dv [S,B,H,v]; cos/sin [S,1,1,emb] from one angle tensor."""
    seq, batch, heads, k_dim, v_dim, emb_dim, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(1257)
    dk = torch.randn((seq, batch, heads, k_dim + emb_dim), dtype=dtype, generator=generator)
    dv = torch.randn((seq, batch, heads, v_dim), dtype=dtype, generator=generator)
    angle = torch.randn((seq, 1, 1, emb_dim), dtype=dtype, generator=generator)
    return dict(dk=dk, dv=dv, cos=angle.cos().contiguous(),
                sin=angle.sin().contiguous(), emb_dim=emb_dim, k_dim=k_dim, v_dim=v_dim)


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_megatron_mla_rope_kv_split_backward(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_megatron_mla_rope_kv_split_backward(**inputs)


def compare(ref_out, res_out, config):
    seq, batch, heads, k_dim, v_dim, emb_dim, dtype = config
    assert isinstance(ref_out, tuple) and len(ref_out) == GRADIENT_COUNT, \
        f"reference returned {len(ref_out)} values, expected {GRADIENT_COUNT}"
    assert isinstance(res_out, tuple) and len(res_out) == GRADIENT_COUNT, \
        f"candidate returned {type(res_out).__name__} of length {len(res_out)}"
    assert all(value is None for value in ref_out[2:]), "reference must pad with None"
    assert all(value is None for value in res_out[2:]), \
        "candidate must return nine trailing None gradients"
    rtol, atol = TOLERANCE[dtype]
    expected_shapes = ((seq, batch, heads, k_dim + v_dim), (seq, batch, 1, emb_dim))
    for index, shape in enumerate(expected_shapes):
        expected, actual = ref_out[index], res_out[index]
        assert isinstance(actual, torch.Tensor), f"gradient {index} is not a tensor"
        assert actual.shape == expected.shape == shape, \
            f"gradient {index} shape {tuple(actual.shape)} != {shape}"
        assert actual.dtype == expected.dtype == dtype
        assert actual.device == expected.device
        torch.testing.assert_close(actual.float(), expected.float(),
                                   rtol=rtol, atol=atol)


def compute_ref_performance(**inputs):
    return checks.measure(kernelgenbench.baseline.v1_megatron_mla_rope_kv_split_backward,
                          inputs, kernelgenbench.device)


def compute_res_performance(**inputs):
    return checks.measure(kernelgenbench.solution.v1_megatron_mla_rope_kv_split_backward,
                          inputs, kernelgenbench.device)


@label("v1_megatron_mla_rope_kv_split_backward")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_mla_rope_kv_split_backward(config):
    seq, batch, heads, k_dim, v_dim, emb_dim, dtype = config
    assert emb_dim % 4 == 0  # upstream forward precondition
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    assert res_inputs["dk"].shape == (seq, batch, heads, k_dim + emb_dim)
    assert res_inputs["dv"].shape == (seq, batch, heads, v_dim)
    assert res_inputs["cos"].shape == (seq, 1, 1, emb_dim)
    assert res_inputs["cos"].is_contiguous() and res_inputs["sin"].is_contiguous()

    before = {name: value.clone() for name, value in res_inputs.items()
              if isinstance(value, torch.Tensor)}
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config)

    for name, value in before.items():
        assert torch.equal(value, res_inputs[name]), f"input {name} was modified"
        for produced in res_out[:2]:
            assert not torch._C._is_alias_of(produced, res_inputs[name]), \
                f"gradient aliases {name}"

    # The embedding gradient must reduce over all heads; the data below makes
    # that observable (head 0 alone differs from the head sum), which the
    # candidate negative in the verification script relies on.
    if heads > 1:
        trailing = res_inputs["dk"][..., k_dim:].float()
        assert not torch.allclose(trailing.sum(2, keepdim=True), trailing[:, :, :1]), \
            "test data cannot distinguish a head reduction from head 0"

    return checks.timings(compute_ref_performance(**inputs),
                          compute_res_performance(**inputs))
