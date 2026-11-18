"""No-bias forward via VectorAddOp PyTorch fallback; result has the dtype of residual + attention_output."""
# Copyright (c) Microsoft Corporation. SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/transformer/inference/op_binding/residual_add.py#L53-L73
# Source: https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/transformer/inference/op_binding/vector_add.py#L21-L28
import torch


def v1_ds_attention_residual_no_bias(hidden_state, residual, attention_output, mp_size):
    summed = residual + attention_output
    return ((1.0 / mp_size) * summed.float() + hidden_state.float()).to(summed.dtype)
