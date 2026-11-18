"""SGLang routed-ID expansion with a local shared expert.

Official source:
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/moe/waterfill.py#L33-L68

``expand_topk_with_shared_expert`` remaps routed ids from the old per-rank
layout to the new ``old_experts_per_rank + 1`` layout, appends the local shared
expert id ``source_rank * (old_epr + 1) + old_epr``, zeroes weights of negative
(dropped) routes and appends ``shared_weight`` only for rows with at least one
valid route.  A row with no valid route gets the ``-1`` sentinel and weight 0.
Output ids keep the input id dtype; output weights keep the input weight dtype.
Inputs are not modified.
"""
import torch


def v1_expand_shared_expert_topk(
    topk_ids, topk_weights, num_routed_experts, world_size, source_rank, shared_weight
):
    per_rank = num_routed_experts // world_size
    valid = topk_ids >= 0
    mapped = torch.where(
        valid, topk_ids + torch.div(topk_ids, per_rank, rounding_mode="floor"), topk_ids
    )
    present = valid.any(dim=1, keepdim=True)
    shared = torch.where(
        present, source_rank * (per_rank + 1) + per_rank, -1
    ).to(topk_ids.dtype)
    weights = torch.where(valid, topk_weights, 0)
    extra = torch.where(present, shared_weight, 0.0).to(topk_weights.dtype)
    return torch.cat((mapped, shared), dim=1), torch.cat((weights, extra), dim=1)
