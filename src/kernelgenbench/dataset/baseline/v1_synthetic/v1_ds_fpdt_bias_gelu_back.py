"""DeepSpeed FPDT unreduced gradient with bias already included in x."""
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/sequence/fpdt_layer.py#L1171-L1176
# Copyright (c) Microsoft Corporation. SPDX-License-Identifier: Apache-2.0
import torch


def v1_ds_fpdt_bias_gelu_back(g, x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    factor = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x))
    return (factor + 0.5 * (1 + tanh_out)) * g
