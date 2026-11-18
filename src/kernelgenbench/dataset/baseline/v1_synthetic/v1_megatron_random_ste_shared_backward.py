"""Megatron RandomSTEShared backward: identity gradient plus two ``None`` slots.

Official source:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/moe_utils.py#L1245-L1278

The forward signature is ``(logits, std, layer_number)``, so autograd expects a
three-tuple.  The gradient object is returned unchanged; the two scalar
arguments receive ``None`` gradients.
"""


def v1_megatron_random_ste_shared_backward(grad_output):
    return grad_output, None, None
