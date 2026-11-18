"""Megatron fused vocabulary-parallel cross-entropy backward.

Official implementation:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_cross_entropy.py#L124-L134

Same arithmetic as the unfused backward (the saved softmax is rewritten in
place, the target entry is decremented and the result is scaled by
``grad_output``), but the JIT-fused helper returns a *new* BF16 tensor cast
from the mutated FP32 softmax.  No label smoothing upstream.  Because the
returned dtype and the aliasing differ from the unfused entry this is a
separate Item.
"""
import torch


def v1_megatron_vocab_parallel_ce_backward_fused(softmax, target_mask, masked_target,
                                                 grad_output):
    n = softmax.shape[-1]
    grad = softmax.view(-1, n)
    rows = torch.arange(masked_target.numel(), device=softmax.device)
    keep = 1.0 - target_mask.view(-1).float()
    grad[rows, masked_target] -= keep
    softmax.mul_(grad_output.unsqueeze(-1))
    # The fused helper casts its in-place FP32 result to BF16 before returning.
    return softmax.to(torch.bfloat16)
