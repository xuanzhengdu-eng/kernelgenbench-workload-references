"""Megatron SwiGLU forward: silu(first half) * second half.

Official source (pinned commit 266f1c97dca477ca8b92c16087506da2000b0b84):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_bias_swiglu.py#L15-L26

y [..., 2H] must have an even last width; torch.chunk(2, -1) then gives two
H-wide halves (the official helper's shapes broadcast only for even widths). The upstream helper is
wrapped by @jit_fuser; verification keeps that decorator.
"""
import torch
import torch.nn.functional as F


def v1_swiglu(y):
    assert y.shape[-1] % 2 == 0, "SwiGLU requires an even 2H width"
    y_1, y_2 = torch.chunk(y, 2, -1)
    return F.silu(y_1) * y_2
