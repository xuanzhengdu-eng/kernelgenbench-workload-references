# Source-checked synthetic-workload benchmark reference.
"""OP-abd21066927b: Megatron IndicesToMultihot forward.

Official source:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_indices_converter.py#L176-L287

The official implementation is a CUDA Triton kernel. Contract covered here:
indices and probs_indices are [num_tokens, topk]; the result is a bool
[num_tokens, num_of_local_experts] multihot and a probs tensor of the same
shape/dtype as probs_indices. An entry equal to -1 or >= num_of_local_experts
is ignored. The upstream kernel does not guard other negative ids and has no
defined duplicate-index contract, so the tested domain keeps every row's ids
unique and never passes an id below -1.
"""
import torch


def v1_indices_to_multihot(indices, probs_indices, num_of_local_experts):
    assert indices.dim() == 2 and indices.shape == probs_indices.shape
    assert num_of_local_experts > 0
    assert bool((indices >= -1).all()), "official kernel only defines -1 as the padding id"
    num_tokens, topk = indices.shape
    present = torch.zeros(
        (num_tokens, num_of_local_experts), dtype=torch.bool, device=indices.device
    )
    probs = torch.zeros(
        (num_tokens, num_of_local_experts),
        dtype=probs_indices.dtype,
        device=probs_indices.device,
    )
    ids = indices.to(torch.int64)
    rows = torch.arange(num_tokens, device=indices.device).unsqueeze(1)
    # The kernel stores one (token, topk) pair at a time; keep the same order.
    for slot in range(topk):
        slot_ids = ids[:, slot]
        valid = (slot_ids != -1) & (slot_ids < num_of_local_experts)
        present[rows[:, 0][valid], slot_ids[valid]] = True
        probs[rows[:, 0][valid], slot_ids[valid]] = probs_indices[valid, slot]
    return present, probs
