"""DeepSpeed explicit backward averaging TP-folded gradients.
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/moe/ep_tp_dispatch.py#L398-L422
Averages equal-shape gradients across ranks, returns a fresh contiguous tensor
and None for the group argument; does not mutate the original gradient.
"""
import torch
import torch.distributed as dist

def v1_ds_average_gradient_backward(grad_output,group):
    local=grad_output.contiguous();parts=[torch.empty_like(local) for _ in range(group.size())]
    dist.all_gather(parts,local,group=group)
    return torch.stack(parts).sum(0).div_(group.size()),None
