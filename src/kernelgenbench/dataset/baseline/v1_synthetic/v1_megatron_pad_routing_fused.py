"""Megatron fused routing-map padding; fresh int32 output.

Source (pinned official commit):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_pad_routing_map.py#L31-L100

The official entry is decorated with ``jit_fuser`` and launches a Triton kernel
over the transposed ``[num_experts, num_tokens]`` map.  It returns a fresh
int32 tensor in ``[num_tokens, num_experts]`` order without modifying the
input.  For ``num_tokens == 0`` it returns the input object unchanged.
"""
import torch


def v1_megatron_pad_routing_fused(routing_map, pad_multiple):
    num_tokens, _ = routing_map.shape
    if num_tokens == 0:
        return routing_map
    work = routing_map.transpose(0, 1).contiguous().int()
    num_to_pad = (-work.sum(dim=1)) % pad_multiple
    zero_ranks = torch.cumsum((work == 0).int(), dim=1)
    mask = (zero_ranks <= num_to_pad.unsqueeze(1)) & (work == 0)
    return torch.where(mask, 1, work).transpose(0, 1)
