"""Megatron global gradient p-norm over plain tensors with a real collective.

https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/optimizer/clip_grads.py#L55-L142
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/utils.py#L1178-L1186

Contract (plain CUDA tensors, no DTensor/FSDP, no Transformer Engine/Apex, so the
local multi-tensor fallback is the compiled-in path):
  * A single tensor is wrapped into a one-element list; an empty list is allowed
    only on the L2 branch, where it yields 0.0.
  * ``norm_type == inf`` uses the maximum absolute value, reduced with a real MAX
    all-reduce over ``grad_stats_parallel_group`` on a scalar that upstream always
    allocates on CUDA (``device='cuda'``), so an empty list fails in ``max()``
    first.
  * ``norm_type == 2`` nests the per-tensor norms exactly like
    ``local_multi_tensor_l2_norm`` (``norm([[norm(g) for g in grads]])``), squares
    the fp32 result on CUDA, SUM all-reduces it and returns ``item() ** 0.5``; an
    empty list therefore yields 0.0. Measured: the CUDA scalar also reduces
    correctly when the gradient tensors themselves are CPU tensors.
  * Every other p starts from the Python float 0.0, adds ``norm(g, p) ** p`` in
    fp32, SUM all-reduces, and returns ``item() ** (1/p)``; with an empty list the
    accumulator stays a float and ``all_reduce`` raises TypeError, as upstream
    does.
  * The reduction is over the caller's group, so every rank returns the same
    scalar; inputs are never modified. The DTensor data-parallel pre-reduction and
    the tensor-valued return used by some callers are out of scope here.
"""
import torch
import torch.distributed as dist


def v1_megatron_grad_norm(grads_for_norm, norm_type, grad_stats_parallel_group):
    if isinstance(grads_for_norm, torch.Tensor):
        grads_for_norm = [grads_for_norm]
    p = float(norm_type)
    if p == float('inf'):
        local = max(grad.abs().max() for grad in grads_for_norm)
        total = torch.tensor([float(local)], dtype=torch.float32, device='cuda')
        dist.all_reduce(total, op=dist.ReduceOp.MAX, group=grad_stats_parallel_group)
        return total[0].item()
    if p == 2.0:
        if grads_for_norm:
            nested = [[torch.norm(grad) for grad in grads_for_norm]]
            reduced = torch.norm(torch.tensor(nested))
            total = torch.tensor([float(reduced)], dtype=torch.float32, device='cuda') ** p
        else:
            total = torch.zeros(1, dtype=torch.float32, device='cuda') ** p
        dist.all_reduce(total, op=dist.ReduceOp.SUM, group=grad_stats_parallel_group)
        return total.item() ** (1.0 / p)
    total = 0.0
    for grad in grads_for_norm:
        total = total + torch.norm(grad, p) ** p
    dist.all_reduce(total, op=dist.ReduceOp.SUM, group=grad_stats_parallel_group)
    return total.item() ** (1.0 / p)
