"""Megatron RandomSTE forward: rank-seeded random router logits.

Official source:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/moe_utils.py#L1231-L1242
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/moe_utils.py#L1194-L1230

``apply_random_logits`` runs ``RandomSTE.apply``; its forward clones ``logits``
and fills the clone with N(0, 1) draws from the expert-parallel RNG tracker.
The input tensor is not modified, the result is fresh storage with the same
shape, dtype and device, and autograd passes the gradient through unchanged.
The tracker state is modelled here with an explicit integer seed: for a
freshly seeded CUDA generator, ``manual_seed(seed)`` produces the same state as
the tracker's ``add(name, seed)``.
"""
import torch


class _RandomSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, expert_seed):
        generator = torch.Generator(device=logits.device)
        generator.manual_seed(int(expert_seed))
        return torch.empty_like(logits).normal_(generator=generator)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def v1_megatron_random_ste(logits, expert_seed):
    return _RandomSTE.apply(logits, int(expert_seed))
