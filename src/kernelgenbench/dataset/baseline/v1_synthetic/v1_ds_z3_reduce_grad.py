# Source (limited scope): https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/compile/z3.cpp#L252-L339
# Scope: reduce-scatter, dtype conversion and accumulation adapter; NCCL async scheduling excluded.
# Source (limited scope): https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/compile/z3.cpp#L51-L66
# Scope: endBackward state transition only; actual reduce-scatter source is cited separately.
"""DeepSpeed Z3 reduce-scatter gradient at endBackward boundary, one ds_id.

`grad` has exactly world_size * grad_buffer.numel() flattened elements in rank
order. Communication uses grad dtype; the local reduced shard is cast to the
buffer dtype before copying or accumulating. Even an empty shard sets the
accumulation flag. endBackward(update=True) clears the flag and invalidates the
parameter. Return (zero-length gradient result, new_acc_flag, new_valid_flag).
The grad buffer is mutated; grad is read-only. Scope excludes reload buffers,
expected_grad_dtype override, async scheduling and profiling. Groups run matching
collective calls with contiguous grad buffers; finite FP32/FP16/BF16 inputs.
"""
import torch

def v1_ds_z3_reduce_grad(grad,grad_buffer,has_acc_grad,param_valid,update,group=None,pre_div_reduce=False):
    world=torch.distributed.get_world_size(group) if torch.distributed.is_initialized() else 1
    send=grad.contiguous().view(-1).clone()
    if pre_div_reduce:send.div_(world)
    if torch.distributed.is_initialized():
        recv=torch.empty(grad_buffer.numel(),dtype=grad.dtype,device=grad.device)
        torch.distributed.reduce_scatter_tensor(recv,send,group=group)
    else:recv=send
    if not pre_div_reduce:recv.div_(world)
    if grad_buffer.numel():
        shard=recv.to(grad_buffer.dtype).view(grad_buffer.shape)
        if has_acc_grad:grad_buffer.add_(shard)
        else:grad_buffer.copy_(shard)
    return grad.new_empty((0,)),False if update else True,False if update else param_valid
