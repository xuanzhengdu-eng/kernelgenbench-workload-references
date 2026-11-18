"""Megatron frozen-weight linear backward.

Official source (NVIDIA Megatron-LM, pinned commit):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/layers.py#L327-L359

Contract:
  * ``weight`` is frozen (no weight or bias gradient is produced).
  * The forward is ``inp @ weight.T + bias``; the backward returns
    ``grad_input = grad_output @ weight`` and four ``None`` entries for
    ``weight``, ``bias``, ``allreduce_dgrad`` and ``tp_group``:
    ``(grad_input, None, None, None, None)``.
  * ``grad_input`` keeps ``grad_output``'s dtype; inputs are not modified.
"""
import torch


def v1_router_linear_back_frozen(grad_output, weight):
    assert grad_output.ndim >= 2 and weight.ndim == 2
    assert grad_output.shape[-1] == weight.shape[0]
    grad_input = grad_output.matmul(weight)
    return grad_input, None, None, None, None
