"""vLLM expert alignment and indirection buffers.

Official implementation:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fused_moe/moe_align_block_size.py#L55-L169

``routing_map`` is ``(tokens, topk)`` int32/int64.  Invalid ids (outside
``[0, num_experts)``) are skipped.  With ``expert_map`` and
``ignore_invalid_experts=False`` the kernel first counts every valid id and
then maps block expert ids through ``expert_map`` (so an empty block maps
``-1`` to the last entry).  ``pad_sorted_ids`` rounds the sorted length to a
block multiple.  The optional ``scatter_idx`` is ``(tokens*topk, 1)`` and holds
the routed row or -1.
"""
import torch


def v1_megatron_moe_align_graphable(routing_map, block_size, num_local_experts,
                                    valid_tokens, expert_map=None,
                                    ignore_invalid_experts=False,
                                    pad_sorted_ids=False,
                                    return_scatter_idx=False):
    assert routing_map.ndim == 2
    assert routing_map.dtype in (torch.int32, torch.int64)
    assert block_size > 0 and num_local_experts > 0
    n, topk = routing_map.shape
    experts = num_local_experts
    sentinel = n * topk
    assert int(valid_tokens) == n
    if return_scatter_idx and expert_map is not None and not ignore_invalid_experts:
        raise ValueError('scatter_idx with expert_map requires ignore_invalid_experts')
    size = sentinel + experts * (block_size - 1)
    if pad_sorted_ids:
        size = (size + block_size - 1) // block_size * block_size
    if sentinel < experts:
        size = min(sentinel * block_size, size)
    sorted_ids = torch.full((size,), sentinel, dtype=torch.int32, device=routing_map.device)
    expert_ids = torch.full(((size + block_size - 1) // block_size,), -1,
                            dtype=torch.int32, device=routing_map.device)
    flat = routing_map.reshape(-1)
    valid = (flat >= 0) & (flat < experts)
    safe = flat.clamp(0, experts - 1).long()
    groups = flat.clone()
    if expert_map is not None and ignore_invalid_experts:
        mapped = expert_map[safe]
        valid = valid & (mapped >= 0) & (mapped < experts)
        groups = mapped
    offset = 0
    for expert in range(experts):
        pairs = torch.nonzero(valid & (groups == expert), as_tuple=False).flatten().int()
        count = pairs.numel()
        padded = (count + block_size - 1) // block_size * block_size
        if count:
            sorted_ids[offset:offset + count] = pairs
            expert_ids[offset // block_size:(offset + padded) // block_size] = expert
        offset += padded
    if expert_map is not None and not ignore_invalid_experts:
        expert_ids = expert_map[expert_ids.long()].int()
    result = (sorted_ids, expert_ids,
              torch.tensor([offset], dtype=torch.int32, device=routing_map.device))
    if return_scatter_idx:
        scatter = torch.where(valid,
                              torch.arange(sentinel, device=routing_map.device, dtype=torch.int32),
                              -1).reshape(-1, 1)
        result += (scatter,)
    return result
