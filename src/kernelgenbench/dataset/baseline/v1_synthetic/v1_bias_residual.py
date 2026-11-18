"""Spatial three-input FP16 kernel; contiguous [B,S,C] or channels-last [N,C,H,W], fresh output."""
# Copyright (c) Microsoft Corporation. SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/spatial/csrc/pt_binding.cpp#L58-L79
# Source: https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/spatial/csrc/opt_bias_add.cu#L50-L78
# Source: https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/spatial/csrc/opt_bias_add.cu#L117-L144
# Source: https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/spatial/csrc/pt_binding.cpp#L12-L32
import torch


@torch.no_grad()
def v1_bias_residual(activation, bias, other):
    if activation.ndim == 3 and activation.is_contiguous() and other.is_contiguous():
        channels = activation.shape[-1]
        channel_shape = (1, 1, channels)
    elif activation.ndim == 4 and all(
        x.is_contiguous(memory_format=torch.channels_last) for x in (activation, other)
    ):
        channels = activation.shape[1]
        channel_shape = (1, channels, 1, 1)
    else:
        raise ValueError("expected contiguous [B,S,C] or channels-last [N,C,H,W]")
    if activation.dtype != torch.float16 or other.shape != activation.shape or other.dtype != activation.dtype:
        raise ValueError("activation and other must have matching fp16 shapes")
    if bias.shape != (channels,) or bias.dtype != torch.float16 or not bias.is_contiguous():
        raise ValueError("bias must be a contiguous fp16 channel vector")
    if activation.numel() == 0 or channels % 8:
        raise ValueError("expected nonempty input and channels divisible by 8")
    if activation.device != bias.device or activation.device != other.device:
        raise ValueError("all tensors must be on the same device")
    return activation + (bias.view(channel_shape) + other)
