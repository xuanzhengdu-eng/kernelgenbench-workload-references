# Source-checked synthetic-workload benchmark reference.
"""OP-f09a2aa29254: Megatron inference local expert token count.

Official source:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/inference/moe/permute.py#L60-L75

Contract: routing_map is the flattened [num_tokens, topk] expert assignment;
experts outside [local_expert_start, local_expert_start + num_local_experts)
are ignored, including negative ids. The result is an int32 tensor of shape
[num_local_experts] on the routing_map device. The old fork-only valid_tokens
and persistent parameters do not exist upstream and are not part of this
reference.
"""
import torch


def v1_count_local_expert_tokens(routing_map, local_expert_start, num_local_experts):
    assert routing_map.dim() >= 1
    assert num_local_experts > 0
    local = routing_map.to(torch.int64) - int(local_expert_start)
    valid = (local >= 0) & (local < num_local_experts)
    counts = torch.bincount(local[valid].reshape(-1), minlength=num_local_experts)
    return counts.to(torch.int32)
