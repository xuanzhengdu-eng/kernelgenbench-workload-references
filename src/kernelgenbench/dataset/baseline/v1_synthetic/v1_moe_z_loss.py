# Fixed official source: https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/moe_utils.py#L149-L178
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Derived from Megatron-LM. License: tests/megatron_official/upstream/LICENSE
"""Unfused Megatron MoE router z-loss, fixed commit 266f1c97dca477ca8b92c16087506da2000b0b84.

The mask follows upstream implementation: True denotes padding. No distributed
loss scaling, router state, communication, or Transformer Engine is included.
"""
import torch


def z_loss_func(logits, z_loss_coeff, padding_mask=None):
    """Return scalar router z-loss; logits [tokens, experts], mask [tokens]."""
    logsum = torch.logsumexp(logits, dim=-1)
    values = torch.square(logsum)
    if padding_mask is not None:
        valid = ~padding_mask
        values = values * valid
        num_valid = valid.sum()
        return values.sum() / torch.clamp(num_valid, min=1.0) * z_loss_coeff
    return torch.mean(values) * z_loss_coeff



def v1_moe_z_loss(logits, z_loss_coeff, padding_mask):
    return z_loss_func(logits, z_loss_coeff, padding_mask)
