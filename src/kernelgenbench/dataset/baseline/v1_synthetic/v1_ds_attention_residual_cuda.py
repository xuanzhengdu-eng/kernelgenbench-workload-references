"""DeepSpeed CUDA inference epilogue; stores into residual and returns its alias."""
# Copyright (c) Microsoft Corporation. SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/transformer/inference/csrc/pt_binding.cpp#L1815-L1851
# Source: https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/transformer/inference/csrc/gelu.cu#L145-L289
# Source: https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/transformer/inference/csrc/gelu.cu#L300-L423
import torch


@torch.no_grad()
def v1_ds_attention_residual_cuda(hidden_state, residual, attention_output, attention_bias, final_bias, mp_size, mlp_after_attn, add_attn_bias, pre_attn_norm):
    values = (hidden_state, residual, attention_output, attention_bias, final_bias)
    if any(x.dtype != hidden_state.dtype or x.device != hidden_state.device for x in values):
        raise ValueError("all tensors must share dtype and device")
    if any(not x.is_contiguous() for x in values):
        raise ValueError("the pointer kernel requires contiguous tensors")
    if hidden_state.ndim != 3 or residual.shape != hidden_state.shape or attention_output.shape != hidden_state.shape:
        raise ValueError("hidden_state, residual and attention_output must have matching [B,S,H] shapes")
    if attention_bias.shape != (hidden_state.shape[2],) or final_bias.shape != attention_bias.shape:
        raise ValueError("both biases must match the hidden dimension")
    if hidden_state.numel() == 0 or mp_size < 1:
        raise ValueError("expected nonempty inputs and a positive parallel size")
    if hidden_state.dtype not in (torch.float32, torch.float16, torch.bfloat16) or hidden_state.shape[-1] % 4:
        raise ValueError("native four-element vectors require aligned fp32/fp16/bf16 inputs")
    h, r, attn, ab, fb = (value.float() for value in values)
    if mlp_after_attn:
        if pre_attn_norm:
            out = torch.addcmul(h, r + attn + fb + ab, h.new_tensor(1.0 / mp_size))
        else:
            out = r + h + fb
    else:
        if add_attn_bias:
            r = r + ab
        out = torch.addcmul(h + attn, r + fb, h.new_tensor(1.0 / mp_size))
    residual.copy_(out)
    return residual
