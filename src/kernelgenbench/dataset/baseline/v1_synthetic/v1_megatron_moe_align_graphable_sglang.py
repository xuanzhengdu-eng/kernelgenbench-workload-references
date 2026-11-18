"""SGLang expert alignment with the invalid-expert bucket.

Official implementation:
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py#L32-L202

``routing_map`` is ``(tokens, topk)`` int32/int64.  SGLang counts an extra
``-1`` invalid-expert bucket, so the padded size is
``tokens*topk + (num_experts+1)*(block_size-1)`` and is rounded to a multiple
of four.  ``ignore_invalid_experts`` drops the ``-1`` bucket entirely.  The
returned active length counts padded slots, not tokens.
"""
import torch


def v1_megatron_moe_align_graphable_sglang(routing_map, block_size, num_local_experts,
                                           valid_tokens, ignore_invalid_experts=False):
    n, topk = routing_map.shape
    experts = num_local_experts
    assert routing_map.ndim == 2
    assert routing_map.dtype in (torch.int32, torch.int64)
    assert block_size > 0 and experts > 0
    assert int(valid_tokens) == n
    sentinel = n * topk
    size = sentinel * block_size if sentinel < experts + 1 else sentinel + (experts + 1) * (block_size - 1)
    size = (size + 3) // 4 * 4
    sorted_ids = torch.full((size,), sentinel, dtype=torch.int32, device=routing_map.device)
    expert_ids = torch.full(((size + block_size - 1) // block_size,), -1,
                            dtype=torch.int32, device=routing_map.device)
    flat = routing_map.reshape(-1)
    offset = 0
    for expert in range(-1, experts):
        if expert == -1 and ignore_invalid_experts:
            continue
        pairs = torch.nonzero(flat == expert, as_tuple=False).flatten().int()
        count = pairs.numel()
        padded = (count + block_size - 1) // block_size * block_size
        if count:
            sorted_ids[offset:offset + count] = pairs
            expert_ids[offset // block_size:(offset + padded) // block_size] = expert
        offset += padded
    return sorted_ids, expert_ids, torch.tensor([offset], dtype=torch.int32,
                                                device=routing_map.device)
