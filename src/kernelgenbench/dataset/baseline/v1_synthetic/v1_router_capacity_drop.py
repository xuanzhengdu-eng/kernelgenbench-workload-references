# Fixed-source anchor for the documented functional sub-contract: https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/moe_utils.py#L877-L940
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Derived from Megatron-LM. License: tests/megatron_official/upstream/LICENSE
"""Official Megatron MoE router capacity calculation and token dropping.

NVIDIA/Megatron-LM@266f1c97dca477ca8b92c16087506da2000b0b84:
megatron/core/transformer/moe/moe_utils.py:206-224, 877-940.
"""
import math
import torch


def get_capacity(num_tokens, num_experts, capacity_factor, min_capacity=None):
    capacity = math.ceil((num_tokens / num_experts) * capacity_factor)
    if min_capacity is not None and capacity < min_capacity:
        capacity = min_capacity
    return capacity


def apply_router_token_dropping(routing_probs, routing_map, router_topk, capacity_factor,
                                drop_policy="probs", pad_to_capacity=False):
    assert routing_probs.ndim == 2 and routing_map.ndim == 2
    num_tokens, num_experts = routing_probs.shape
    expert_capacity = get_capacity(
        num_tokens=num_tokens * router_topk,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
    )
    if expert_capacity > num_tokens:
        capacity_mask = torch.ones_like(routing_probs).bool()
    else:
        if drop_policy == "probs":
            _, capacity_indices = torch.topk(routing_probs, k=expert_capacity, dim=0, sorted=False)
            capacity_mask = torch.zeros_like(routing_probs).scatter(0, capacity_indices, 1).bool()
        elif drop_policy == "position":
            _, capacity_indices = torch.topk(routing_map.int(), k=expert_capacity, dim=0, sorted=False)
            capacity_mask = torch.zeros_like(routing_probs).scatter(0, capacity_indices, 1).bool()
        else:
            raise ValueError(f"Invalid drop_policy: {drop_policy}")
    if pad_to_capacity:
        final_map = capacity_mask
        final_probs = routing_probs * final_map
    else:
        final_map = torch.logical_and(routing_map, capacity_mask)
        final_probs = routing_probs * final_map
    return final_probs, final_map


def v1_router_capacity_drop(routing_probs, routing_map, router_topk, capacity_factor, pad_to_capacity, drop_policy):
    return apply_router_token_dropping(routing_probs, routing_map, router_topk, capacity_factor, drop_policy, pad_to_capacity)
