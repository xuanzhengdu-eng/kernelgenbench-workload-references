"""Megatron GatedDeltaNet decay and update gates (official pinned source).

Official source (Megatron-LM 266f1c97), GatedDeltaNet.forward:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/ssm/gated_delta_net.py#L369-L370

Contract: `g = -exp(A_log) * softplus(alpha.float() + dt_bias)` is computed in
float32 regardless of the alpha dtype (alpha is upcast, dt_bias is broadcast),
and `beta` is the sigmoid kept in beta's own dtype. A_log/dt_bias are the
per-value-head parameters, so they are 1-D vectors of length num_value_heads.
"""
import torch
import torch.nn.functional as F


def v1_gdn_decay_gate(A_log_local_cp, dt_bias_local_cp, alpha, beta):
    g = -A_log_local_cp.exp() * F.softplus(alpha.float() + dt_bias_local_cp)
    return g, beta.sigmoid()
