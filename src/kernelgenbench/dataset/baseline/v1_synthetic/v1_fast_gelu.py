# v1_fast_gelu: Megatron fast_gelu (native (tanh)); 723-row 13.
"""Megatron fast_gelu reference, transcribed from the pinned body.

Source (pinned commit):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/activations.py#L20-L23

Contract: tanh approximation with constants 0.7978845608 and 0.044715 in that order. Bitwise equal to vLLM FastGELU and to DeepSpeed FPDT bias_gelu on all tested dtypes, so one item carries all three references.
"""
import torch
def v1_fast_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))
