"""Megatron backward of a sequence-parallel gather: reduce-scatter the gradient.

Official source (NVIDIA/Megatron-LM, pinned commit 266f1c97):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/mappings.py#L296-L348

Scalar/group sub-contract of
``_GatherFromSequenceParallelRegion.backward`` with
``tensor_parallel_output_grad=True``: the summed gradient is reduce-scattered
along the first axis and this rank keeps its contiguous segment.  Kept
behaviour:
  * ``group.size() == 1`` returns the input object unchanged (official bypass),
  * the first axis must divide by the group size,
  * the returned 5-tuple ``(grad, None, None, None, None)`` matches the official
    backward's one gradient plus four non-tensor slots,
  * the shard is freshly allocated, so it never aliases the incoming gradient.
"""
import torch
import torch.distributed as dist


def v1_megatron_gather_reduce_scatter_backward(grad_output, group):
    world = group.size()
    if world == 1:
        return grad_output, None, None, None, None
    assert grad_output.shape[0] % world == 0, (
        "First dimension of the tensor should be divisible by tensor parallel size"
    )
    total = grad_output.contiguous().clone()
    dist.all_reduce(total, group=group)
    size = total.shape[0] // world
    out = total.narrow(0, group.rank() * size, size).clone(memory_format=torch.contiguous_format)
    return out, None, None, None, None
