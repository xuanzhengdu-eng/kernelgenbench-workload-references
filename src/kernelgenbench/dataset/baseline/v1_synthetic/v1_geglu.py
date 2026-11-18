# v1_geglu: Megatron geglu (native (tanh)); 723-row 16.
"""Megatron geglu reference, transcribed from the pinned body.

Source (pinned commit):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_bias_geglu.py#L16-L27

Contract: torch.chunk split on the last axis, so an odd width splits into unequal halves and the two halves broadcast instead of intersecting: width 3 returns width 2, whereas vLLM/SGLang return width 1 with different values.
"""
import torch
def v1_geglu(y):
    y_1, y_2 = torch.chunk(y, 2, -1)
    return (y_1 * 0.5 * (1.0 + torch.tanh(0.79788456 * y_1 * (1 + 0.044715 * y_1 * y_1)))) * y_2
