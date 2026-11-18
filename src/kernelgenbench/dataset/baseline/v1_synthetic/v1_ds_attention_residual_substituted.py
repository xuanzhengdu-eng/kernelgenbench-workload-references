"""Post-norm ResidualAddOp.forward with a supplied replacement buffer and PyTorch fallback."""
# Copyright (c) Microsoft Corporation. SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/transformer/inference/op_binding/residual_add.py#L29-L73
import torch


def v1_ds_attention_residual_substituted(hidden_state, residual, attention_output, attention_bias, final_bias, residual_add, mp_size, mlp_after_attn, add_attn_bias):
    if mlp_after_attn:
        out = residual_add.float() + hidden_state.float() + final_bias.float()
    else:
        hidden_state.add_(attention_output + (residual_add + final_bias) / mp_size)
        if add_attn_bias:
            hidden_state.add_(attention_bias / mp_size)
        out = hidden_state
    residual_add.copy_(out.to(hidden_state.dtype))
    return residual_add
