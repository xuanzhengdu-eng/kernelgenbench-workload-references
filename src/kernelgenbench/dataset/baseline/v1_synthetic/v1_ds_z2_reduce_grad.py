# Source (limited scope): https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/compile/z2.cpp#L33-L86
# Exact range: endBackward L33-L40, blank L41, flushReduceBucket L42-L86 (NCCL all-reduce,
# slice copy/accumulate into the grad buffer, has_acc_grad update, performCleanup).
# Scope: endBackward and all-reduce copy/accumulate adapter; actual NCCL async scheduling excluded.
"""DeepSpeed ZeRO-2 gradient reduction after endBackward synchronization.

For one ds_id, average grad across ranks every step, then copy its flattened
slice into grad_buffer if has_acc_grad=False or add it otherwise. Nonempty
buffers set state=True; `update=True` resets the state after the flush.
Return an empty tensor (dc::reduce_grad wrapper) and next accumulation flag.
grad is read-only; grad_buffer is mutated. Scope is FP32 normal operation,
contiguous buffer, identical collective order and no expected dtype override;
all buffers/offsets are valid. NCCL async scheduling is outside this ABI.
"""
import torch

def v1_ds_z2_reduce_grad(grad,grad_buffer,offset,has_acc_grad,update,group=None,pre_div_reduce=False):
    send=grad.contiguous().view(-1).clone();world=torch.distributed.get_world_size(group) if torch.distributed.is_initialized() else 1
    if pre_div_reduce:send.div_(world)
    if torch.distributed.is_initialized():torch.distributed.all_reduce(send,group=group)
    if not pre_div_reduce:send.div_(world)
    next_acc=has_acc_grad
    if grad_buffer.numel():
        part=send[offset:offset+grad_buffer.numel()]
        if has_acc_grad:grad_buffer.view(-1).add_(part)
        else:grad_buffer.view(-1).copy_(part)
        next_acc=True
    if update:next_acc=False
    return grad.new_empty((0,)),next_acc
