"""Megatron weighted SwiGLU, token-wise weighted.

Official source (pinned commit):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_bias_swiglu.py#L45-L48

Contract: ``y`` has shape ``(..., 2H)``; the first half is the gate passed
through SiLU, the second half is the linear value.  ``weights`` broadcasts over
the hidden axis and the result keeps ``y.dtype`` and has shape ``(..., H)``.
The official helper is wrapped by ``@jit_fuser``.
"""
import torch
import torch.nn.functional as F


def _swiglu(y):
    y_1, y_2 = torch.chunk(y, 2, -1)
    return F.silu(y_1) * y_2


def v1_weighted_swiglu(y, weights):
    dtype = y.dtype
    res = _swiglu(y) * weights
    return res.to(dtype)
