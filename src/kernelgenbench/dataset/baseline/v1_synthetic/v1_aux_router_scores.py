# Fixed official source: https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/moe_utils.py#L826-L874
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Derived from Megatron-LM. License: tests/megatron_official/upstream/LICENSE
"""Pure PyTorch Megatron auxiliary router scores (fixed-source equivalent).

This implements only the ``fused=False`` branch of
``compute_routing_scores_for_aux_loss`` from Megatron-LM commit
266f1c97dca477ca8b92c16087506da2000b0b84.  See the accompanying documentation
for the important padding-mask convention and limitations.
"""

import torch


def compute_routing_scores_for_aux_loss(
    logits, topk, score_function, fused=False, padding_mask=None
):
    """Return (boolean routing_map, float32 scores), both [tokens, experts].

    ``padding_mask`` is a 1-D boolean tensor where True denotes padding, not
    validity. Invalid inputs are deliberately left to the same PyTorch
    operations as the unfused upstream implementation.
    """
    if fused:
        raise ValueError("Only fused=False is supported by this reference")
    if score_function == "softmax":
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits.to(torch.float32))
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    _, top_indices = torch.topk(scores, k=topk, dim=1)
    routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
    if padding_mask is not None:
        valid_mask = (~padding_mask).unsqueeze(-1)
        routing_map = routing_map * valid_mask
        scores = scores * valid_mask
    return routing_map, scores



def v1_aux_router_scores(logits, topk, score_function, padding_mask):
    return compute_routing_scores_for_aux_loss(logits, topk, score_function, fused=False, padding_mask=padding_mask)
