"""Megatron MLA fused RoPE forward for the query, SBHD / CP1 subset.

Official source (pinned commit), entry point:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_mla_yarn_rope_apply.py#L327-L376
Official source (pinned commit), Triton forward kernel:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_mla_yarn_rope_apply.py#L68-L147

The upstream function rewrites the last ``emb_dim`` columns of ``q`` in place:
it reads the even/odd pairs as ``[x1, x2]`` and writes the split halves
``[x1*cos_left - x2*sin_left, x2*cos_right + x1*sin_right]``.  The first
``qk_head_dim`` columns are untouched.  Only the SBHD layout, CP size one and
``rotary_interleaved=False`` are covered: the Triton entry asserts
``not rotary_interleaved`` and the fork-only interleaved-output branch has no
official counterpart.
"""
import torch


def v1_megatron_mla_rope_inplace_(
    q,
    cos,
    sin,
    qk_head_dim,
    emb_dim,
    cu_seqlens_q=None,
    cp_rank=0,
    cp_size=1,
    rotary_interleaved=False,
):
    assert q.ndim == 4 and q.shape[-1] == qk_head_dim + emb_dim
    assert emb_dim % 4 == 0 and qk_head_dim > 0
    assert cu_seqlens_q is None, "only the SBHD layout is covered"
    assert cp_rank == 0 and cp_size == 1, "only the CP1 subset is covered"
    assert not rotary_interleaved, "the pinned kernel rejects interleaved output"
    assert cos.shape == sin.shape and cos.shape[-1] == emb_dim
    assert cos.shape[0] >= q.shape[0]
    assert q.stride(-1) == 1
    assert cos.is_contiguous() and sin.is_contiguous()

    seq = q.shape[0]
    half = emb_dim // 2
    c = cos[:seq].float()
    s = sin[:seq].float()
    x = q[..., qk_head_dim:].float()
    a = x[..., ::2]
    b = x[..., 1::2]
    left = a * c[..., :half] - b * s[..., :half]
    right = b * c[..., half:] + a * s[..., half:]
    q[..., qk_head_dim:].copy_(torch.cat((left, right), -1).to(q.dtype))
    return q
