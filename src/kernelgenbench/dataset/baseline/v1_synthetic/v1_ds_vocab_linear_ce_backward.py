"""DeepSpeed vocab-parallel linear cross entropy backward, official autograd contract.

https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/sequence/linear_cross_entropy.py#L168-L212

The reference reproduces the upstream analytic backward of
``vocab_parallel_linear_cross_entropy``:
* ``dlogits = softmax(logits) * (grad_loss / sum_exp)`` and subtract
  ``grad_loss`` at each valid target owned by this rank;
* ``grad_hidden = dlogits @ weight``, ``grad_weight = dlogits.T @ hidden`` and
  ``grad_bias = dlogits.sum(0)``;
* ``grad_hidden`` is summed across the TP group because each rank only saw its
  vocabulary shard;
* ``reduction='mean'`` divides the upstream ``grad_loss`` by the number of
  valid tokens, matching the upstream mean over valid tokens;
* ignored targets contribute no gradient.

Inputs are FP32; this reference targets the torch fallback path, not the
Triton/DeepGEMM fused kernels (which the pinned module only enables when the
mixed-precision GEMM dispatch exists).
"""
import torch
import torch.distributed as dist

from .v1_ds_vocab_linear_ce import _world_size, vocab_ce_stats


def v1_ds_vocab_linear_ce_backward(
    hidden,
    weight,
    bias,
    target,
    group,
    vocab_start,
    grad_loss,
    ignore_index=-100,
    reduction="none",
):
    exps, den, _, valid, in_shard, local = vocab_ce_stats(
        hidden, weight, bias, target, group, vocab_start, ignore_index
    )
    if reduction == "mean":
        grad_loss = grad_loss / valid.sum().clamp_min(1)
    elif reduction not in ("sum", "none"):
        raise ValueError("unsupported reduction")
    grad = exps / den[:, None]
    rows = torch.arange(target.numel(), device=target.device)
    grad[rows[in_shard], local[in_shard]] -= 1
    grad *= torch.where(valid, grad_loss, torch.zeros_like(grad_loss))[:, None]
    gh = grad @ weight
    gw = grad.T @ hidden
    gb = grad.sum(0) if bias is not None else None
    if _world_size(group) > 1:
        dist.all_reduce(gh, op=dist.ReduceOp.SUM, group=group)
    return gh, gw, gb
