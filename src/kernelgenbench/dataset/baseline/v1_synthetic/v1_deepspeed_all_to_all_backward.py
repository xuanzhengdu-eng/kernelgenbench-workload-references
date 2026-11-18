"""Reference: DeepSpeed all-to-all backward (two-element tuple).

Official implementation (pinned DeepSpeed commit
``8af59b2e7a91c227081617240e98aa282eadc5f9``):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/moe/sharded_moe.py#L97-L109

DeepSpeed's ``_AllToAll`` autograd function takes two inputs ``(group, input)``,
so its backward returns exactly ``(None, grad)``: the group is not
differentiable and the gradient is the same equal-split all-to-all applied to
the incoming gradient (the exchange is its own inverse for an equal split).  The
result is always a fresh tensor, including the single-rank case, which is the
observable difference from the Megatron backward.
"""
import torch
import torch.distributed as dist


def v1_deepspeed_all_to_all_backward(group, grad_output):
    assert group is not None
    local = grad_output.contiguous()
    grad = torch.empty_like(local)
    dist.all_to_all_single(grad, local, group=group)
    return (None, grad)
