"""Megatron attention output gate: x * sigmoid(gate.float()) cast to x.dtype.

Official source (pinned commit 266f1c97dca477ca8b92c16087506da2000b0b84):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/attention.py#L1226-L1233

Unlike the packed GLU item, this entry takes the value and gate as two separate
tensors. gate is made contiguous and reshaped to x.shape, the sigmoid is
evaluated in float32, and the product is cast back to the value dtype. The
kernel is wrapped by Megatron's @jit_fuser; the verification keeps that
decorator and binds Attention._apply_output_gate directly.
"""
import torch


def v1_megatron_attention_output_gate(x, gate):
    x_dtype = x.dtype
    gate = gate.contiguous().view(*x.shape)
    return (x * torch.sigmoid(gate.float())).to(x_dtype)
