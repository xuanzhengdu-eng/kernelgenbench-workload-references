"""Megatron RandomSTE backward: the straight-through gradient is the identity.

Official source:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/moe_utils.py#L1194-L1230

``RandomSTE.backward`` returns ``grad_output`` itself, preserving the object,
shape, dtype, device and strides.  It does not clone or rescale the gradient.
"""


def v1_megatron_random_ste_backward(grad_output):
    return grad_output
