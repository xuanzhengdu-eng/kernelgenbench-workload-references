"""vLLM ViT torch SDPA wrapper, layout bshd_vit.

Official entry: ``torch_sdpa_wrapper`` in vllm-project/vllm
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/attention/ops/vit_attn_wrappers.py#L244-L302

Q [B,S,Hq,D], K [B,S,Hkv,D], V [B,S,Hkv,Dv]. With cu_seqlens, the sequence axis is
split into contiguous segments and each segment is attended independently; otherwise
the whole batch is attended at once. enable_gqa repeats K/V when Hq != Hkv. scale may
be None (torch default 1/sqrt(D)). Output is [B,S,Hq,Dv] in the input dtype and
contiguous. This is the former ``v1_dense_attention_unified`` bshd_vit branch.
"""
import math
import torch


def v1_dense_attention_bshd_vit(query, key, value, scale=None, cu_seqlens=None, enable_gqa=False):
    assert query.ndim == key.ndim == value.ndim == 4
    assert query.shape[:2] == key.shape[:2] == value.shape[:2]
    assert key.shape[2] == value.shape[2] and query.shape[-1] == key.shape[-1]
    assert query.shape[2] % key.shape[2] == 0
    if not enable_gqa:
        assert query.shape[2] == key.shape[2]
    starts = [0, query.shape[1]] if cu_seqlens is None else cu_seqlens.tolist()
    assert starts[0] == 0 and starts[-1] == query.shape[1]
    assert all(a < b for a, b in zip(starts, starts[1:]))
    ratio = query.shape[2] // key.shape[2]
    factor = scale if scale is not None else query.shape[-1] ** -0.5
    pieces = []
    for start, end in zip(starts, starts[1:]):
        q = query[:, start:end].float().permute(0, 2, 1, 3)
        k = key[:, start:end].float().repeat_interleave(ratio, dim=2).permute(0, 2, 1, 3)
        v = value[:, start:end].float().repeat_interleave(ratio, dim=2).permute(0, 2, 1, 3)
        pieces.append(torch.matmul((q @ k.transpose(-1, -2) * factor).softmax(-1), v).permute(0, 2, 1, 3))
    return torch.cat(pieces, dim=1).to(query.dtype).contiguous()
