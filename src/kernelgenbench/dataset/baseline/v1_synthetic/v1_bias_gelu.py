"""Megatron bias plus tanh-approximate GELU helper."""
# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_bias_gelu.py#L16-L19
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import torch


def v1_bias_gelu(bias, y):
    x = bias + y
    return x * 0.5 * (1 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))
