# Derived from DeepSpeed, Copyright Microsoft Corporation. SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/transformer/inference/op_binding/gated_activation.py#L25-L37
"""DeepSpeed GATED_SILU=4 fallback with rank-3 activations and a channel bias. The second half is the FP32 gate."""
import torch
import torch.nn.functional as F


def v1_ds_bias_swiglu_gate_second(activation, bias, activation_func_type):
    if activation.ndim != 3 or activation_func_type != 4:
        raise ValueError("expected rank-3 activations and GATED_SILU=4")
    activations = activation + bias.reshape(1, 1, -1)
    hidden, gate = activations.chunk(2, dim=-1)
    return hidden * F.silu(gate.float()).to(activations.dtype)
