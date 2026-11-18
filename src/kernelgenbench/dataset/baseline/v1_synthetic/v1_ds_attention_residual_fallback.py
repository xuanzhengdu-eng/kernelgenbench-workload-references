"""PyTorch ResidualAddOp.residual_add_fallback; the GPT-J branch also writes hidden_state."""
# Copyright (c) Microsoft Corporation. SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/transformer/inference/op_binding/residual_add.py#L29-L51
import torch


def v1_ds_attention_residual_fallback(hidden_state, residual, attention_output, attention_bias, final_bias, mp_size, mlp_after_attn, add_attn_bias, pre_attn_norm):
    if mlp_after_attn:
        if pre_attn_norm:
            summed = residual.float() + attention_output.float() + attention_bias.float() + final_bias.float()
            out = summed / mp_size + hidden_state.float()
        else:
            out = residual.float() + hidden_state.float() + final_bias.float()
    else:
        hidden_state.add_(attention_output + (residual + final_bias) / mp_size)
        if add_attn_bias:
            hidden_state.add_(attention_bias / mp_size)
        out = hidden_state
    residual.copy_(out.to(hidden_state.dtype))
    return residual
