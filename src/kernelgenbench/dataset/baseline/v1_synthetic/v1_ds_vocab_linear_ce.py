"""DeepSpeed vocab-parallel linear cross entropy, official fused forward contract.

https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/sequence/linear_cross_entropy.py#L215-L298

This reference models the FP32 single-shard / TP-shard contract of
``vocab_parallel_linear_cross_entropy``:
* ``logits = hidden @ weight.T + bias`` in FP32, with ``weight`` holding this
  rank's vocabulary shard and ``target`` holding global vocabulary ids;
* a global row max is reduced (MAX) across the TP group, then shifted
  exponentials and their sum are reduced (SUM);
* the selected target logit is taken only on the rank that owns the id and is
  reduced (SUM) across the group;
* ``loss = log(sum_exp) - target_logit`` for valid targets, zero at
  ``ignore_index``;
* ``reduction`` is ``none`` (per-token), ``sum`` or ``mean`` over valid tokens.

The upstream entry also strips ignored tokens before the projection; this
reference keeps them and writes zero loss, which is the same observable
contract for ``reduction='none'``.  Gradient computation lives in
``v1_ds_vocab_linear_ce_backward``.
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F


def _world_size(group):
    if group is None and not dist.is_initialized():
        return 1
    return dist.get_world_size(group)


def vocab_ce_stats(hidden, weight, bias, target, group, vocab_start, ignore_index):
    logits = F.linear(hidden, weight, bias).float()
    maxval = logits.amax(-1)
    if _world_size(group) > 1:
        dist.all_reduce(maxval, op=dist.ReduceOp.MAX, group=group)
    shifted = logits - maxval[:, None]
    exps = shifted.exp()
    den = exps.sum(-1)
    valid = target != ignore_index
    local = (target - vocab_start).clamp(0, weight.shape[0] - 1).long()
    in_shard = valid & (target >= vocab_start) & (target < vocab_start + weight.shape[0])
    selected = torch.where(
        in_shard, shifted.gather(1, local[:, None]).squeeze(1), torch.zeros_like(den)
    )
    if _world_size(group) > 1:
        dist.all_reduce(den, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(selected, op=dist.ReduceOp.SUM, group=group)
    return exps, den, selected, valid, in_shard, local


def v1_ds_vocab_linear_ce(
    hidden,
    weight,
    bias,
    target,
    group,
    vocab_start,
    ignore_index=-100,
    reduction="none",
):
    exps, den, selected, valid, _, _ = vocab_ce_stats(
        hidden, weight, bias, target, group, vocab_start, ignore_index
    )
    loss = torch.where(valid, den.log() - selected, torch.zeros_like(den))
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.sum() / valid.sum().clamp_min(1)
    raise ValueError("unsupported reduction")
