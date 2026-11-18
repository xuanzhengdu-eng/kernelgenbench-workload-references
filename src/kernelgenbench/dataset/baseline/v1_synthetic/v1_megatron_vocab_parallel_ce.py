"""Megatron vocabulary-parallel cross entropy, single-shard forward.

Official implementation:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/cross_entropy.py#L122-L217
Entry: https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/cross_entropy.py#L219-L231

Contract (single tensor-parallel shard, so the local statistics are global):
* ``logits`` is the local ``[..., local_vocab]`` shard and ``target`` holds the
  *global* vocabulary ids; ids outside ``[0, local_vocab)`` are masked (no
  target logit, loss still computed).
* The FP32 working copy is mutated in place to the local softmax.  When the
  input is FP32 that working copy aliases the caller's tensor, so the input is
  observably rewritten; FP16/BF16 inputs stay untouched because ``.float()``
  makes a copy.
* The loss is FP32 and keeps the leading shape of ``target``.
* ``label_smoothing`` follows the upstream formula
  ``(1 - s) * loss - s * mean(log_softmax)`` with ``s = alpha * V / (V - 1)``.

The fused entry (``fused_vocab_parallel_cross_entropy``) computes the same
values and is a separate reference of the same Item.
"""
import torch


def v1_megatron_vocab_parallel_ce(logits, target, label_smoothing):
    x = logits.float()
    local_max = x.max(dim=-1).values
    # In-place subtraction mirrors the upstream implementation.
    x -= local_max.unsqueeze(-1)
    n = x.shape[-1]
    assert label_smoothing < 1.0

    # Single shard starts at 0, so only ids outside [0, local_vocab) are masked.
    keep = (target < 0) | (target >= n)
    index = target.clone()
    index[keep] = 0
    rows = torch.arange(target.numel(), device=x.device)
    picked = x.view(-1, n)[rows, index.view(-1)].clone().view_as(target)
    picked[keep] = 0.0

    x.exp_()
    denominator = x.sum(dim=-1)
    loss = denominator.log() - picked
    x.div_(denominator.unsqueeze(-1))

    if label_smoothing > 0:
        smoothing = label_smoothing * n / (n - 1)
        loss = (1.0 - smoothing) * loss - smoothing * x.log().mean(dim=-1)
    return loss
