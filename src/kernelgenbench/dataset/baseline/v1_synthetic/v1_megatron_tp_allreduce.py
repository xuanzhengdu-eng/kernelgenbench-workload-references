"""Megatron tensor-parallel all-reduce forward.

``_ReduceFromModelParallelRegion.apply`` calls ``_reduce``: a one-rank group
returns the input untouched, otherwise ``torch.distributed.all_reduce`` runs on
``input_.contiguous()`` and the *original* handle is returned.  A dense input is
therefore reduced in place; a non-contiguous input keeps its values because the
reduced temporary is discarded.  The wrapper's backward is the identity.
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/mappings.py#L22-L31
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/mappings.py#L217-L233
"""
import torch
import torch.distributed as dist

class _TPReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,group):
        assert group is not None
        if group.size()>1:dist.all_reduce(x.contiguous(),group=group)
        return x
    @staticmethod
    def backward(ctx,grad):return grad,None

def v1_megatron_tp_allreduce(x,group):
    return _TPReduce.apply(x,group)
