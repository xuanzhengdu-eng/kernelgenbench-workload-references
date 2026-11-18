"""Megatron fp32 gradient zero-count, official NVIDIA entry.

https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/optimizer/clip_grads.py#L195-L263

Filters gradients that are missing, shared, or replicated by tensor model
parallelism, accumulates ``numel - count_nonzero`` on a CUDA int64 scalar and
SUM-reduces it over ``grad_stats_parallel_group``; the result is a Python int.
The legacy reference came from the flagos-ai/Megatron-LM-FL fork.  The official
version differs in the FSDP branch and in the TP-duplicate test (attribute first,
then ``tp_group.rank()``), so those are kept here.  ``tp_group`` is required for
this scoped item; the upstream ``tp_group=None`` fallback reads Megatron's global
TP state, which is outside a single-rank harness.
"""
import torch
import torch.distributed as dist

def v1_megatron_count_grad_zeros(parameters, grad_stats_parallel_group, use_decoupled_grad=False,
                                 tp_group=None):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    assert tp_group is not None
    total_num_zeros = torch.zeros(1, dtype=torch.int64, device="cuda")
    for param in parameters:
        if getattr(param, "__fsdp_param__", False) and param.grad is not None:
            grad = param.grad._local_tensor
            total_num_zeros = (grad.numel() - torch.count_nonzero(grad)) + total_num_zeros
            continue
        grad_attr = "decoupled_grad" if use_decoupled_grad else "grad"
        grad = getattr(param, grad_attr, None)
        is_not_shared = not getattr(param, "shared", False)
        is_not_tp_duplicate = (getattr(param, "tensor_model_parallel", False)
                               or tp_group.rank() == 0)
        if grad is not None and is_not_shared and is_not_tp_duplicate:
            grad = grad.detach()
            num_zeros = grad.numel() - torch.count_nonzero(grad)
            total_num_zeros = num_zeros + total_num_zeros
    dist.all_reduce(total_num_zeros, op=dist.ReduceOp.SUM, group=grad_stats_parallel_group)
    return total_num_zeros.item()
