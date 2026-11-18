"""Megatron MLA fused RoPE backward for the query, SBHD / CP1 subset.

Official source (pinned commit), the explicit transpose map:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_mla_yarn_rope_apply.py#L270-L326
Official source (pinned commit), Triton backward kernel:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_mla_yarn_rope_apply.py#L148-L209

This is ``ApplyMLARotaryEmbQ.backward``: the incoming gradient is rewritten in
place on the last ``emb_dim`` columns.  The forward writes split halves
(``[left, right]``); the backward reads those halves and writes interleaved pairs
(``[x1[0], x2[0], x1[1], x2[1], ...]``).  The first ``qk_head_dim`` columns pass
through unchanged.  The upstream path has no ``inverse`` switch and no
interleaved-output switch; those fork-only branches are not part of this
contract.  ``backward`` returns the gradient followed by eight ``None`` entries.
"""
import torch


def v1_megatron_mla_rope_backward_(
    grad,
    cos,
    sin,
    qk_head_dim,
    emb_dim,
    cu_seqlens_q=None,
    cp_rank=0,
    cp_size=1,
):
    assert grad.ndim == 4 and grad.shape[-1] == qk_head_dim + emb_dim
    assert emb_dim % 4 == 0 and qk_head_dim > 0
    assert cu_seqlens_q is None, "only the SBHD layout is covered"
    assert cp_rank == 0 and cp_size == 1, "only the CP1 subset is covered"
    assert cos.shape == sin.shape and cos.shape[-1] == emb_dim
    assert cos.shape[0] >= grad.shape[0]
    assert grad.stride(-1) == 1
    assert cos.is_contiguous() and sin.is_contiguous()

    seq = grad.shape[0]
    half = emb_dim // 2
    c = cos[:seq].float()
    s = sin[:seq].float()
    x = grad[..., qk_head_dim:].float()
    left, right = x.chunk(2, -1)
    out = torch.stack(
        (
            left * c[..., :half] + right * s[..., half:],
            -left * s[..., :half] + right * c[..., half:],
        ),
        -1,
    ).flatten(-2)
    grad[..., qk_head_dim:].copy_(out.to(grad.dtype))
    return (grad,) + (None,) * 8
