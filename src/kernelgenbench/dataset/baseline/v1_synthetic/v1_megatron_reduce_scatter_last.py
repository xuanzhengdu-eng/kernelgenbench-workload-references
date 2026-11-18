"""Megatron forward SUM reduce-scatter along the last axis.

``_reduce_scatter_along_last_dim`` reshapes to ``[-1, last]``, moves the last-axis
chunks to dim 0 with split/cat, reduce-scatters, and reshapes to the shard shape.
Because the split/cat step materializes a new tensor, even a one-rank group returns
a fresh tensor rather than the input object.
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/mappings.py#L99-L112
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/mappings.py#L155-L213
"""
import torch
import torch.distributed as dist

def v1_megatron_reduce_scatter_last(input_,group):
    world=group.size()
    assert input_.shape[-1]%world==0
    target=list(input_.shape);target[-1]//=world
    flat=input_.reshape(-1,input_.shape[-1]);chunk=flat.shape[-1]//world
    concat=torch.cat(torch.split(flat,chunk,dim=1),dim=0)
    if world==1:return concat.reshape(target)
    out=concat.new_empty((concat.shape[0]//world,concat.shape[1]))
    dist.reduce_scatter_tensor(out,concat.contiguous(),group=group)
    return out.reshape(target)
