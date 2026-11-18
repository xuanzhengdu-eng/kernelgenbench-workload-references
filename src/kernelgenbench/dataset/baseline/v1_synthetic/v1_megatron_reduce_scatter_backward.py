"""Megatron backward of sequence-parallel reduce-scatter: first-dim all-gather.

``_ReduceScatterToSequenceParallelRegion.backward`` forwards to
``_gather_along_first_dim``: a one-rank group returns the gradient object itself,
otherwise a fresh dim-0 concatenation in rank order is allocated.  Four slots:
gradient, group, ``input_split_sizes`` and ``use_global_buffer``.
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/mappings.py#L114-L151
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/mappings.py#L351-L397
"""
import torch
import torch.distributed as dist

def _gather_along_first_dim(input_,group):
    world=group.size()
    if world==1:return input_
    shape=list(input_.size());shape[0]*=world
    output=torch.empty(shape,dtype=input_.dtype,device=input_.device)
    dist.all_gather_into_tensor(output,input_.contiguous(),group=group)
    return output

def v1_megatron_reduce_scatter_backward(grad_output,group):
    return _gather_along_first_dim(grad_output,group),None,None,None
