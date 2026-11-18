"""Single-rank Megatron shared-expert MLP: gated or plain SwiGLU.

Official source:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/mlp.py#L246-L354
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/shared_experts.py#L121-L129

Contract (``MLP.forward`` eager branch used by ``SharedExpertMLP.forward``):
- ``x`` is ``[..., hidden]``; ``w1`` is ``[2*ffn, hidden]``, ``w2`` is
  ``[hidden, ffn]``; the FC1 output is split into gate/linear halves along the
  last dimension.
- Optional ``b1`` is added to the FC1 output. ``clamp_value`` clamps the gate
  half from above and the linear half on both sides before the activation.
- ``silu(gate) * (linear + linear_offset)`` is projected by ``w2``; when
  ``gate_weight`` is given the result is multiplied by
  ``sigmoid(linear(x, gate_weight))`` (the shared-expert gate).
- ``SharedExpertMLP`` asserts ``add_bias_linear is False``, so the gated shared
  expert has no FC1 bias; the base ``MLP`` path does support ``b1``.
- Returns a fresh tensor; no input is modified.
"""
import torch
import torch.nn.functional as F


def v1_megatron_shared_expert_mlp(x, w1, w2, b1, gate_weight, linear_offset, clamp_value):
    projected = F.linear(x, w1)
    if b1 is not None:
        projected = projected + b1
    u, v = projected.chunk(2, dim=-1)
    if clamp_value is not None:
        u = u.clamp(max=clamp_value)
        v = v.clamp(min=-clamp_value, max=clamp_value)
    y = F.linear(F.silu(u) * (v + linear_offset), w2)
    if gate_weight is not None:
        y = y * torch.sigmoid(F.linear(x, gate_weight))
    return y
