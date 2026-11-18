"""Megatron MLA fused KV split and rotary embedding, SBHD / CP1 subset.

Official source (pinned commit), entry point:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_mla_yarn_rope_apply.py#L735-L782
Official source (pinned commit), Triton kernel that defines the arithmetic:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_mla_yarn_rope_apply.py#L379-L487

The upstream function splits ``kv`` into ``k`` (the first ``k_dim`` columns) and
``v`` (the last ``v_dim`` columns), rotates the separate ``k_pos_emb`` tail and
concatenates it after ``k``.  Only the SBHD layout, CP size one and
``rotary_interleaved=False`` are covered here: the Triton entry asserts
``not rotary_interleaved`` and the old fork-only interleaved output branch has no
official counterpart, so it is not part of this contract.  ``value`` is a fresh
contiguous tensor, not a view into ``kv``.
"""
import torch


def v1_megatron_mla_rope_kv_split(
    kv,
    k_pos_emb,
    cos,
    sin,
    emb_dim,
    k_dim,
    v_dim,
    cu_seqlens_kv=None,
    cp_rank=0,
    cp_size=1,
    rotary_interleaved=False,
):
    assert kv.ndim == 4 and kv.shape[-1] == k_dim + v_dim
    assert k_pos_emb.shape == (*kv.shape[:2], 1, emb_dim)
    assert emb_dim % 4 == 0 and k_dim > 0 and v_dim > 0
    assert cu_seqlens_kv is None, "only the SBHD layout is covered"
    assert cp_rank == 0 and cp_size == 1, "only the CP1 subset is covered"
    assert not rotary_interleaved, "the pinned kernel rejects interleaved output"
    assert cos.shape == sin.shape and cos.shape[-1] == emb_dim
    assert cos.shape[0] >= kv.shape[0]
    assert kv.stride(-1) == 1 and k_pos_emb.stride(-1) == 1
    assert cos.is_contiguous() and sin.is_contiguous()

    seq = kv.shape[0]
    half = emb_dim // 2
    x = k_pos_emb.float()
    c = cos[:seq].float()
    s = sin[:seq].float()
    a = x[..., ::2]
    b = x[..., 1::2]
    left = a * c[..., :half] - b * s[..., :half]
    right = b * c[..., half:] + a * s[..., half:]
    rotation = torch.cat((left, right), -1).to(kv.dtype).expand(*kv.shape[:3], emb_dim)

    key = torch.cat((kv[..., :k_dim], rotation), -1)
    value = kv[..., k_dim:].clone(memory_format=torch.contiguous_format)
    return key, value
