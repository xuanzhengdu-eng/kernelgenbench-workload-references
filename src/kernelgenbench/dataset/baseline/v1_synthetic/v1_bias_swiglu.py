# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Source: https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_bias_swiglu.py#L15-L41
"""Apply biased SwiGLU with the gate in the first half of the last dimension."""
import torch
import torch.nn.functional as F


def v1_bias_swiglu(y, bias):
    gate, value = (y + bias).chunk(2, dim=-1)
    return F.silu(gate) * value
