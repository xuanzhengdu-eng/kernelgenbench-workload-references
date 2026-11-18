"""DeepSpeed CUDABiasActivation configured as IDENTITY, mutates activation and returns None."""
# Copyright (c) Microsoft Corporation. SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/v2/kernels/core_ops/bias_activations/bias_activation.py#L21-L62
# Source: https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/v2/kernels/core_ops/bias_activations/bias_activation_cuda.cu#L16-L20
# Source: https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/v2/kernels/core_ops/bias_activations/bias_activation_cuda.cu#L50-L89
# Source: https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/v2/kernels/core_ops/bias_activations/bias_activation.cpp#L38-L68
import torch


@torch.no_grad()
def v1_bias_add_v2_identity(activation, bias=None):
    if activation.ndim != 2 or not activation.is_contiguous() or activation.numel() == 0 or activation.shape[1] % 8:
        raise ValueError("expected nonempty contiguous [tokens,channels] with channels divisible by 8")
    if activation.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("expected fp16 or bf16")
    if bias is None:
        return None
    if bias.shape != (activation.shape[1],) or bias.dtype != activation.dtype or bias.device != activation.device or not bias.is_contiguous():
        raise ValueError("expected a contiguous channel bias with matching dtype and device")
    activation.copy_(activation.float() + bias.float())
    return None
