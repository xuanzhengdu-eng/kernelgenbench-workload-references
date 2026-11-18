# Fixed official source: https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/moe_utils.py#L1136-L1159
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Derived from Megatron-LM. License: tests/megatron_official/upstream/LICENSE
"""Megatron MoE expert-bias update, after the distributed reduction.

This intentionally excludes all_reduce and caller-side state mutation.
Source: NVIDIA/Megatron-LM@266f1c97dca477ca8b92c16087506da2000b0b84,
megatron/core/transformer/moe/moe_utils.py:1156-1159.
"""
import torch


def expert_bias_update(global_tokens_per_expert, expert_bias, expert_bias_update_rate):
    """Return updated bias from already globally reduced expert counts.

    The last dimension is the expert dimension. Leading dimensions are
    preserved; inputs must have shapes compatible with the upstream expression.
    No distributed communication or caller-owned bias mutation is performed.
    """
    with torch.no_grad():
        average_tokens = global_tokens_per_expert.sum(dim=-1, keepdim=True) / global_tokens_per_expert.shape[-1]
        offset = average_tokens - global_tokens_per_expert
        updated_expert_bias = expert_bias + torch.sign(offset) * expert_bias_update_rate
        return updated_expert_bias


def v1_expert_bias_update_single_rank(tokens_per_expert, expert_bias, expert_bias_update_rate):
    return expert_bias_update(tokens_per_expert, expert_bias, expert_bias_update_rate)
