"""DeepSpeed Triton inference epilogue on contiguous tensors; all inputs are read-only."""
# Copyright (c) Microsoft Corporation. SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/transformer/inference/triton/residual_add.py#L12-L88
import torch


@torch.no_grad()
def v1_ds_attention_residual(hidden_state, residual, attention_output, attention_bias, final_bias, mp_size, mlp_after_attn, add_attn_bias, pre_attn_norm):
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
    if mlp_after_attn:
        if pre_attn_norm:
            return hidden_state + (residual + final_bias + attention_output + attention_bias) / mp_size
        return hidden_state + residual + final_bias
    out = hidden_state + attention_output + (residual + final_bias) / mp_size
    if add_attn_bias:
        out = out + attention_bias / mp_size
    return out
