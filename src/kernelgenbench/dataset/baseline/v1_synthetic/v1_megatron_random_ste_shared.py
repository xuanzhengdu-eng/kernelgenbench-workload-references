"""Megatron RandomSTEShared forward: random logits plus a shared seeded bias.

Official source:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/moe_utils.py#L1279-L1290
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/moe_utils.py#L1245-L1278

``apply_biased_logits(logits, std, layer_number)`` first applies the RandomSTE
forward, then adds a bias drawn from the data-parallel RNG.  A negative ``std``
means "generate once per layer and reuse": the bias is cached under
``layer_number`` when the layer is not ``None``.  ``cached_bias`` is the value
currently stored for that layer (or ``None``); ``cache_out`` is the value that
must be stored after the call.  Positive/zero ``std`` never writes the cache.
The two integer seeds model the expert-parallel and data-parallel tracker
states.  Both autograd Functions pass the gradient through unchanged; the
non-logits arguments receive no gradient.
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


class _SharedBias(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, bias):
        return value + bias

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def v1_megatron_random_ste_shared(logits, std, layer_number, cached_bias, expert_seed, shared_seed):
    value = _RandomSTE.apply(logits, int(expert_seed))

    cache_out = cached_bias
    if std < 0 and layer_number is not None and cached_bias is not None:
        bias = cached_bias
    else:
        shared_generator = torch.Generator(device=logits.device)
        shared_generator.manual_seed(int(shared_seed))
        bias = torch.empty(
            (logits.shape[-1],), device=logits.device, dtype=logits.dtype
        ).normal_(std=abs(float(std)), generator=shared_generator)
        if std < 0 and layer_number is not None:
            cache_out = bias
    return _SharedBias.apply(value, bias), cache_out
