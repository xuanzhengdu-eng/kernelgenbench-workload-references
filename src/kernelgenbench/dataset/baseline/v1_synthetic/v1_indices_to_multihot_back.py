"""Megatron IndicesToMultihot backward with the saved position map.

Official source (pinned commit):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_indices_converter.py#L239-L280
Forward that produces the saved map:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_indices_converter.py#L186-L237

Contract: ``grad_probs_in_multihot`` is [num_tokens, num_local_experts];
``position_map`` is the int32 map saved by the forward pass, where
``position_map[row, expert]`` is the top-k slot that routed to that expert, or
-1 when the expert is absent.  ``topk`` is the forward top-k.  The backward
returns the same four-tuple as the autograd Function: (None, grad_probs, None,
None), where ``grad_probs`` is a new [num_tokens, topk] tensor in the incoming
gradient dtype.  Each valid position_map entry gathers its multihot gradient;
slots without a valid entry stay 0.  Position maps in the contract are unique
per row; inputs are not modified.
"""
import torch


def v1_indices_to_multihot_back(grad_probs_in_multihot, position_map, topk):
    valid = position_map >= 0
    ids = position_map.clamp(min=0).long()
    n = grad_probs_in_multihot.shape[0]
    grad = torch.zeros((n, topk), dtype=grad_probs_in_multihot.dtype, device=grad_probs_in_multihot.device)
    grad.scatter_add_(1, ids, torch.where(valid, grad_probs_in_multihot, torch.zeros_like(grad_probs_in_multihot)))
    return None, grad, None, None
