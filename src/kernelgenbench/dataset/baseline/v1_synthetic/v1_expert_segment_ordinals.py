"""Stable ordinal inside each contiguous expert assignment segment.

Official source:
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/moe/ep_tp_dispatch.py#L43-L54

The result is always int64 and starts at 0 for every run of equal adjacent
expert ids.  For an empty int64 input DeepSpeed returns the input object itself
(``.to(torch.long)`` is a no-op); every non-empty input produces fresh storage.
The input is never modified; a 2-D input is not part of the contract and the
source fails on it.
"""
import torch


def v1_expert_segment_ordinals(expert_indices):
    if expert_indices.numel() == 0:
        return expert_indices.to(torch.long)
    positions = torch.arange(
        expert_indices.numel(), device=expert_indices.device, dtype=torch.long
    )
    segment_start = torch.zeros(
        expert_indices.numel(), device=expert_indices.device, dtype=torch.bool
    )
    segment_start[0] = True
    segment_start[1:] = expert_indices[1:] != expert_indices[:-1]
    starts = torch.where(segment_start, positions, torch.zeros_like(positions))
    starts = torch.cummax(starts, dim=0).values
    return positions - starts
