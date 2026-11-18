"""Megatron forward SUM reduce-scatter along the first axis.

``_reduce_scatter_along_first_dim`` bypasses a one-rank group (returning the input
object), requires the first dimension to divide the group size, and otherwise
reduce-scatters the contiguous input into a freshly allocated shard.
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/mappings.py#L155-L213
"""
import torch
import torch.distributed as dist

def v1_megatron_reduce_scatter(input_,group):
    world=group.size()
    if world==1:return input_
    assert input_.shape[0]%world==0
    output=input_.new_empty((input_.shape[0]//world,)+input_.shape[1:])
    dist.reduce_scatter_tensor(output,input_.contiguous(),group=group)
    return output
