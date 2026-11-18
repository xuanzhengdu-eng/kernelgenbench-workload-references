"""Megatron MLA fused KV split + YARN RoPE, backward (SBHD, CP1 subset).

Official sources (NVIDIA/Megatron-LM pinned commit ``266f1c97``):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_mla_yarn_rope_apply.py#L675-L732
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_mla_yarn_rope_apply.py#L487-L578

Contract
--------
Backward of ``ApplyMLARotaryEmbKV`` for the SBHD layout with ``cu_seqlens_kv=None``
and no context parallelism (``cp_rank=0, cp_size=1``).  ``dk`` is the upstream
gradient of ``o_key`` ``[S, B, H, emb_dim + k_dim]`` and ``dv`` that of
``o_value`` ``[S, B, H, v_dim]``; ``cos``/``sin`` are ``[S, 1, 1, emb_dim]``
contiguous.  The gradient wrt the fused ``kv`` input is
``cat(dk[..., :k_dim], dv)``.  The gradient wrt the positional embedding sums
the trailing ``emb_dim`` lanes of ``dk`` over heads in FP32, rounds the two
halves to the dEMB dtype, applies the inverse rotation
``(l*cos + r*sin, r*cos - l*sin)`` with the two cos/sin halves, and stores the
result interleaved as ``[S, B, 1, emb_dim]`` (even lanes from the first
expression, odd lanes from the second).  The kernel only supports
``rotary_interleaved=False`` (the forward asserts it), so the interleaved
branch of the old reference is removed.  The Triton kernel builds ``tl.arange``
over ``k_dim``/``v_dim``/``emb_dim``, so all three must be powers of two; the
tests only use such sizes.  The backward returns one gradient per
forward argument (11 values), the last nine being ``None``.
"""
import torch


def v1_megatron_mla_rope_kv_split_backward(dk, dv, cos, sin, emb_dim, k_dim, v_dim):
    assert dk.ndim == 4 and dk.shape[-1] == emb_dim + k_dim
    assert dv.shape == (*dk.shape[:3], v_dim)
    dkv = torch.cat((dk[..., :k_dim], dv), -1)
    half = emb_dim // 2
    x = dk[..., k_dim:].float()
    left, right = x.chunk(2, -1)
    # The source rounds the head reduction back to the dEMB dtype before the
    # rotation multiplies.
    left = left.sum(-2, keepdim=True).to(dk.dtype).float()
    right = right.sum(-2, keepdim=True).to(dk.dtype).float()
    c = cos.float()
    s = sin.float()
    even = left * c[..., :half] + right * s[..., half:]
    odd = -left * s[..., :half] + right * c[..., half:]
    demb = torch.stack((even, odd), -1).flatten(-2).to(dk.dtype)
    return (dkv, demb) + (None,) * 9
