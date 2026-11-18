"""Megatron ``_gather_along_last_dim``: all-gather rows, concatenate on the last axis.

Each rank contributes ``input_[None]``-shaped rows; the gathered buffer is chunked
on dim 0 and concatenated on the last axis, giving a contiguous result.  A
one-rank group bypasses the collective and returns the input object itself.
The official body allocates on ``torch.cuda.current_device()``; this reference
allocates on the input device so it is also runnable for CPU Gloo evidence.
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/mappings.py#L80-L96
"""
import torch
import torch.distributed as dist

def v1_megatron_gather_last_dim(input_,group):
    world=group.size()
    if world==1:return input_
    shape=list(input_.size());shape[0]*=world
    output=torch.empty(shape,dtype=input_.dtype,device=input_.device)
    dist.all_gather_into_tensor(output,input_.contiguous(),group=group)
    return torch.cat(output.chunk(world,dim=0),dim=-1).contiguous()
