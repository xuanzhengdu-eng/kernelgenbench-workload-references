# Source (limited scope): https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/compile/z1.cpp#L33-L87
# Scope: one ds_id, synchronized accumulated all-reduce adapter; async scheduling and base executor separate.
"""DeepSpeed ZeRO-1 accumulated gradient update at endBackward boundary.

Tensor ABI for one ds_id: `grad` and optional `accumulated` are same-shaped
FP32 tensors, `grad_buffer` is a contiguous FP32 output shard, offset selects
its position in the flattened gradient. With accumulated=None the saved state
aliases grad; otherwise accumulated is updated in-place with grad. When update
is true, all ranks average the accumulated tensor and copy its selected slice
into grad_buffer, and state is cleared. Return a zero-length tensor and state.
The original operation queues asynchronous work; this ABI includes the
endBackward flush/synchronization. Ranks execute same update order. This scope
excludes profiling skip and expected-grad-dtype override.
"""
import torch

def v1_ds_z1_reduce_grad(grad,accumulated,grad_buffer,offset,update,group=None,pre_div_reduce=False):
    if accumulated is None:state=grad
    else:state=accumulated.add_(grad)
    if update:
        send=state.contiguous().view(-1).clone()
        world=torch.distributed.get_world_size(group) if torch.distributed.is_initialized() else 1
        if pre_div_reduce:send.div_(world)
        if torch.distributed.is_initialized():torch.distributed.all_reduce(send,group=group)
        if not pre_div_reduce:send.div_(world)
        if grad_buffer.numel():grad_buffer.view(-1).copy_(send[offset:offset+grad_buffer.numel()])
        state=None
    return grad.new_empty((0,)),state
