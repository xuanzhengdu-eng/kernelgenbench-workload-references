# Source-checked reference for KernelGenBench row 70 (Megatron branch).
"""Megatron eager full routing with score functions, grouped selection and
optional dense output.

Official sources (NVIDIA fixed commit 266f1c97):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/moe_utils.py#L578-L630
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/moe_utils.py#L671-L813
"""
import torch


def _group_limited_topk(scores, topk, num_tokens, num_experts, num_groups, group_topk):
    group_scores = scores.view(num_tokens, num_groups, -1).topk(topk // group_topk, dim=-1)[0].sum(-1)
    group_idx = torch.topk(group_scores, k=group_topk, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, num_groups, num_experts // num_groups)
        .reshape(num_tokens, -1)
    )
    masked_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
    return torch.topk(masked_scores, k=topk, dim=-1)


def v1_router_score_topk_megatron(logits, topk, use_pre_softmax=False, num_groups=None,
                                  group_topk=None, scaling_factor=None, score_function="softmax",
                                  expert_bias=None, dense_output=False):
    assert logits.dim() == 2, f"Expected 2D logits [num_tokens, num_experts], got {logits.dim()}"
    num_tokens, num_experts = logits.shape

    def compute_topk(scores):
        if group_topk:
            return _group_limited_topk(scores, topk, num_tokens, num_experts,
                                       num_groups, group_topk)
        return torch.topk(scores, k=topk, dim=1)

    if score_function == "softmax":
        if use_pre_softmax:
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            probs, top_indices = compute_topk(scores)
        else:
            scores, top_indices = compute_topk(logits)
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(logits)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits.float()).type_as(logits)
        if expert_bias is not None:
            scores_for_routing = scores + expert_bias
            _, top_indices = compute_topk(scores_for_routing)
            scores = torch.gather(scores, dim=1, index=top_indices).type_as(logits)
        else:
            scores, top_indices = compute_topk(scores)
        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    if scaling_factor:
        probs = probs * scaling_factor

    if dense_output:
        return probs, top_indices

    routing_probs = torch.zeros_like(logits).scatter(1, top_indices, probs)
    routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
    return routing_probs, routing_map
