# Fixed official source: https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/moe_utils.py#L227-L246
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Derived from Megatron-LM. License: tests/megatron_official/upstream/LICENSE
"""Megatron global/local expert token counts with a one-rank identity reduction.

``get_tokens_per_expert_and_token_count`` reduces the local per-expert counts
over the tensor-model-parallel group.  With a single rank that reduction is the
identity, which is the sub-contract exercised here; multi-rank aggregation is
not represented.

With ``with_padding_mask`` the token counts are zero-dimensional float tensors,
otherwise they are Python ints, exactly as returned by
NVIDIA/Megatron-LM@266f1c97dca477ca8b92c16087506da2000b0b84
(megatron/core/transformer/moe/moe_utils.py:227-246).
"""
import torch


def v1_expert_token_counts_single_rank(routing_map, topk, with_padding_mask):
    counts = routing_map.sum(dim=0)
    if with_padding_mask:
        count = counts.sum() / topk
        return counts, count, count.clone()
    n = routing_map.shape[0]
    return counts, n, n
