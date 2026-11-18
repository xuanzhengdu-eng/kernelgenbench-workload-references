"""Megatron unreduced elementwise gradient from bias and input."""
# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_bias_gelu.py#L25-L33
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import torch


def v1_bias_gelu_back(g, bias, y):
    x = bias + y
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    factor = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x))
    return (factor + 0.5 * (1 + tanh_out)) * g
