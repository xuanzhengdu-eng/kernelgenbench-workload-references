# Fixed-source anchor for the documented functional sub-contract: https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/moe_utils.py#L59-L146
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Derived from Megatron-LM. License: tests/megatron_official/upstream/LICENSE
"""Standalone unfused Megatron Switch load-balancing auxiliary loss.

Port of Megatron-LM 266f1c97dca477ca8b92c16087506da2000b0b84,
megatron/core/transformer/moe/moe_utils.py:59-146 (Copyright NVIDIA).
Only the pure torch path is supported; distributed reduction is external.
"""
import torch


def switch_load_balancing_loss_func(
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    total_num_tokens: int,
    topk: int,
    num_experts: int,
    moe_aux_loss_coeff: float,
    fused: bool = False,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the local contribution using externally aggregated counts.

    The optional mask marks valid rows with True; arithmetic, promotion,
    broadcasting and zero-division follow the unfused upstream implementation.
    """
    if padding_mask is not None:
        probs = probs * padding_mask.unsqueeze(-1)
    if fused:
        raise NotImplementedError("Transformer Engine fused auxiliary loss is not supported")
    aggregated_probs_per_expert = probs.sum(dim=0)
    aux_loss = torch.sum(aggregated_probs_per_expert * tokens_per_expert) * (
        num_experts * moe_aux_loss_coeff / (topk * total_num_tokens * total_num_tokens)
    )
    return aux_loss


def v1_moe_switch_aux_loss(probs, tokens_per_expert, total_num_tokens, topk, num_experts, moe_aux_loss_coeff, padding_mask):
    return switch_load_balancing_loss_func(probs, tokens_per_expert, total_num_tokens, topk, num_experts, moe_aux_loss_coeff, fused=False, padding_mask=padding_mask)
