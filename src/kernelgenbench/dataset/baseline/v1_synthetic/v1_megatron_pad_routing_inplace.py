"""Megatron native routing-map padding; mutates the input and returns a view.

Source (pinned official commit):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/moe_utils.py#L636-L693

For each expert column, the first ``(-count) % pad_multiple`` zero entries are
changed to 1.  The operation is in place on the caller's storage; the returned
tensor is the ``transpose(0, 1)`` view of that same storage.  Bool and integer
routing maps are accepted.
"""
import torch


def v1_megatron_pad_routing_inplace(routing_map, pad_multiple):
    work = routing_map.transpose(0, 1)
    num_to_pad = (-work.sum(dim=1)) % pad_multiple
    zero_ranks = torch.cumsum((work == 0).int(), dim=1)
    mask = zero_ranks <= num_to_pad.unsqueeze(1)
    work[mask] = 1
    return work.transpose(0, 1)
