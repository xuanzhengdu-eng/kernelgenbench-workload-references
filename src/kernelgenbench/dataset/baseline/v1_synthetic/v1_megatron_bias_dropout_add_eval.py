# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Source: https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_bias_dropout.py#L11-L59
# Source: https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_bias_dropout.py#L76-L80
"""Megatron inference dropout, aliases x only with matching dtype and all inputs disabling gradients. Rank-3 input and FP32 residual, with optional FP16/BF16 x and bias promoted to FP32."""
import torch
import torch.nn.functional as F


def v1_megatron_bias_dropout_add_eval(x_with_bias, residual, prob):
    x, bias = x_with_bias
    if x.ndim != 3 or x.shape != residual.shape or residual.dtype != torch.float32:
        raise ValueError("expected x and float32 residual with the same 3-D shape")
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("unsupported activation dtype")
    if x.device != residual.device:
        raise ValueError("x and residual must share a device")
    if bias is not None and (bias.shape != (x.shape[-1],) or bias.dtype != x.dtype or bias.device != x.device):
        raise ValueError("bias must be a matching channel vector on the same device")
    if not 0 <= prob < 1:
        raise ValueError("benchmark dropout probability must be in [0, 1)")
    inplace = not x.requires_grad and not residual.requires_grad and (bias is None or not bias.requires_grad)
    if x.dtype != residual.dtype:
        x = x.to(residual.dtype)
        if bias is not None:
            bias = bias.to(residual.dtype)
    if inplace:
        if bias is not None:
            x.add_(bias)
        return x.add_(residual)
    activation = x if bias is None else x + bias
    output = F.dropout(activation, p=prob, training=False)
    return residual + output
