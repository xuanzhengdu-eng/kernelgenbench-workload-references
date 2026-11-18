"""Megatron scaled + optional boolean/causal mask + row softmax (forward).

Official source (torch fallback used whenever the fused kernel is unavailable):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_softmax.py#L299-L350
Official source (fused dispatch and the -10000 mask fill):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_softmax.py#L219-L296
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/utils.py#L49-L51

Scores are scaled in FP32, masked entries are set to -10000, and the softmax is
taken in FP32 before being cast back to the input dtype.
"""
import torch


def v1_scaled_masked_softmax(x, scale, mode, mask=None):
    scores = x.float() * scale
    if mode == 'mask':
        scores = scores.masked_fill(mask, -10000.0)
    elif mode == 'causal':
        s = x.shape[-1]
        assert x.shape[-2] == s
        upper = torch.ones((s, s), device=x.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(upper, -10000.0)
    else:
        assert mode == 'plain'
    return torch.softmax(scores, dim=-1).to(x.dtype)
