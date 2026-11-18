"""Megatron reduce-scatter with FP32 accumulation and an async work handle.

The official helper exchanges equal shards with ``all_to_all_single``, sums them in
FP32 and copies the downcast result into a caller-provided 1-D output.  With
``async_op=True`` it returns a handle whose ``wait`` finishes the sum; otherwise the
sum is completed inline and None is returned.  This reference reproduces the same
numeric contract with ``all_gather`` plus an FP32 sum (equal-size shards make the
two exchange patterns equivalent) and keeps the handle protocol.
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/distributed/reduce_scatter_with_fp32_accumulation.py#L9-L92
"""
import torch
import torch.distributed as dist

class _FP32Work:
    def __init__(self,work,gathered,output,group):
        self.work,self.gathered,self.output,self.group=work,gathered,output,group
    def wait(self):
        if self.work is not None:self.work.wait()
        # Each peer's flat contribution is viewed as (world, shard) and this
        # rank's row is summed in FP32, matching the all-to-all + view(world, -1)
        # accumulation of the pinned source.
        world,shard=self.group.size(),self.output.numel()
        summed=torch.stack([x.reshape(world,shard)[self.group.rank()].float() for x in self.gathered]).sum(0)
        self.output.copy_(summed.reshape_as(self.output))

def v1_megatron_reduce_scatter_fp32_accum(input_,group,output_tensor,async_op=False):
    world=group.size()
    assert output_tensor is not None and output_tensor.ndim==1
    assert input_.numel()==world*output_tensor.numel()
    gathered=[torch.empty_like(input_) for _ in range(world)]
    work=dist.all_gather(gathered,input_.contiguous(),group=group,async_op=async_op)
    handle=_FP32Work(work,gathered,output_tensor,group)
    if async_op:return handle
    handle.wait();return None
