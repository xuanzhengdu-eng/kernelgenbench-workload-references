"""Megatron SwiGLU backward helper: packed input, packed gradient.

Official source (pinned commit 266f1c97dca477ca8b92c16087506da2000b0b84):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_bias_swiglu.py#L54-L69

y [..., 2H] must have an even last width: torch.chunk(2, -1) then gives two
H-wide halves. The official helper fails to broadcast for odd widths.
The returned gradient has y's packed shape: the gate half is
g * sigmoid(a) * (1 + a*(1 - sigmoid(a))) * b and the up half is g * silu(a).
The upstream helper is wrapped by @jit_fuser; verification keeps that decorator.
"""
import torch
import torch.nn.functional as F


def v1_swiglu_back(g, y):
    assert y.shape[-1] % 2 == 0, "SwiGLU backward requires an even 2H width"
    y_1, y_2 = torch.chunk(y, 2, -1)
    return torch.cat(
        (g * torch.sigmoid(y_1) * (1 + y_1 * (1 - torch.sigmoid(y_1))) * y_2,
         g * F.silu(y_1)), -1)
