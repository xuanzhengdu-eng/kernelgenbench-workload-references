"""Megatron fused vocabulary-parallel cross entropy, single-shard forward.

Official implementation:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_cross_entropy.py#L87-L134
Entry: https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_cross_entropy.py#L136-L147

Same loss, mask, in-place FP32 softmax and dtype contract as the unfused entry.
The only difference is that the predicted logit and the sum-exp are packed into
one tensor and reduced with a single SUM all-reduce, and the helpers are
wrapped in ``jit_fuser``; with one shard the packed reduction is the identity,
so the observed values equal the unfused path.  No label smoothing upstream.
This is the second backend reference of the same Item.
"""
import torch


def v1_megatron_vocab_parallel_ce_fused(logits, target, label_smoothing):
    assert label_smoothing == 0, 'the fused upstream entry does not implement label smoothing'
    x = logits.float()
    local_max = x.max(dim=-1).values
    x -= local_max.unsqueeze(-1)
    n = x.shape[-1]

    # Single shard starts at 0, so only ids outside [0, local_vocab) are masked.
    keep = (target < 0) | (target >= n)
    index = target.clone()
    index[keep] = 0
    rows = torch.arange(target.numel(), device=x.device)
    picked = x.view(-1, n)[rows, index.view(-1)].clone().view_as(target)
    picked[keep] = 0.0

    x.exp_()
    denominator = x.sum(dim=-1)
    # pack(picked, denominator) -> one all-reduce -> split; identity for 1 shard
    packed = torch.cat((picked.reshape(-1), denominator.reshape(-1)))
    packed = packed.clone()
    picked = packed[:picked.numel()].view_as(target)
    denominator = packed[picked.numel():].view_as(denominator)
    loss = denominator.log() - picked
    x.div_(denominator.unsqueeze(-1))
    return loss
