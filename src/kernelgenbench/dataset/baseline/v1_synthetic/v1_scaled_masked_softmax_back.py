"""Megatron scaled softmax backward: VJP of the scaled (masked) softmax.

Official source (autograd module; the fused CUDA backward is
``scaled_masked_softmax_cuda.backward``):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_softmax.py#L60-L106
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_softmax.py#L299-L350

The gradient is a NEW tensor: the official ``backward`` returns ``input_grads``
and never writes into the incoming ``output_grads``.  The fork's in-place
writeback branch (plain/causal overwrote ``grad``) has no NVIDIA counterpart and
is removed from this reference; ``mode`` now only selects the forward mask.
"""
import torch


def v1_scaled_masked_softmax_back(output, grad, scale, mode, mask=None):
    y = output.float()
    g = grad.float()
    result = (scale * y * (g - (g * y).sum(dim=-1, keepdim=True))).to(output.dtype)
    return result
