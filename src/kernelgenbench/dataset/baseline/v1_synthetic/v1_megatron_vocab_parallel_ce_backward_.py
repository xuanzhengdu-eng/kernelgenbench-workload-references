"""Megatron unfused vocabulary-parallel cross-entropy backward.

Official implementation:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/cross_entropy.py#L192-L217

``softmax`` is the saved local softmax (FP32) produced by the forward pass; it
is rewritten in place and returned as the gradient tensor (the same object).
``target_mask`` marks rows whose target is outside the shard, ``masked_target``
is the in-shard index and ``grad_output`` is the upstream per-token gradient.
Label smoothing subtracts ``smoothing / V`` from every entry and scales the
target entry by ``1 - smoothing``.
"""
import torch


def v1_megatron_vocab_parallel_ce_backward_(softmax, target_mask, masked_target,
                                            grad_output, label_smoothing=0.0):
    n = softmax.shape[-1]
    grad = softmax.view(-1, n)
    rows = torch.arange(masked_target.numel(), device=softmax.device)
    keep = 1.0 - target_mask.view(-1).float()
    if label_smoothing > 0:
        smoothing = label_smoothing * n / (n - 1)
        grad[rows, masked_target] -= (1.0 - smoothing) * keep
        grad[rows, :] -= smoothing / n
        softmax.mul_(grad_output.unsqueeze(-1))
    else:
        grad[rows, masked_target] -= keep
        softmax.mul_(grad_output.unsqueeze(-1))
    return softmax
