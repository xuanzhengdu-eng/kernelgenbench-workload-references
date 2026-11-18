"""Megatron weighted SwiGLU backward.

Official sources (pinned commit):

* helper: https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_bias_swiglu.py#L90-L97
* wrapper: https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_bias_swiglu.py#L191-L207

Two upstream entry shapes are kept here:

* direct helper ``weighted_swiglu_back(g, y, weights)`` -> ``(grad_y, grad_w)``;
* ``WeightedSwiGLUFunction.backward`` (selected by ``full_backward=True``),
  which restores an FP8 saved input with ``original_dtype`` and returns
  ``(grad_y, grad_w, None)``.

The fork-only clamp / helper-clamped branch is intentionally absent: the fixed
NVIDIA commit has no clamped SwiGLU backward helper.
"""
import torch
import torch.nn.functional as F


def _swiglu(y):
    y_1, y_2 = torch.chunk(y, 2, -1)
    return F.silu(y_1) * y_2


def _swiglu_back(g, y):
    y_1, y_2 = torch.chunk(y, 2, -1)
    sig = torch.sigmoid(y_1)
    return torch.cat(
        (g * sig * (1 + y_1 * (1 - sig)) * y_2, g * F.silu(y_1)), -1
    )


def _weighted_swiglu_back(g, y, weights):
    input_dtype = y.dtype
    w_dtype = weights.dtype
    input_grad = _swiglu_back(g * weights, y)
    # Upstream casts g to the weight dtype before forming the weight gradient.
    weights_grad = _swiglu(y) * g.to(w_dtype)
    weights_grad = torch.sum(weights_grad, dim=-1, keepdim=True)
    return input_grad.to(input_dtype), weights_grad.to(w_dtype)


def v1_weighted_swiglu_back(g, y, weights, full_backward=False,
                            fp8_input_store=False, original_dtype=None):
    if fp8_input_store and (y.dtype != torch.float8_e4m3fn or original_dtype is None):
        raise ValueError("FP8 saved input requires float8_e4m3fn y and original_dtype")
    if full_backward:
        restored = y.to(original_dtype) if fp8_input_store else y
        grad_y, grad_w = _weighted_swiglu_back(g, restored, weights)
        return grad_y, grad_w, None
    if fp8_input_store:
        raise ValueError("FP8 saved state is only valid for full_backward")
    return _weighted_swiglu_back(g, y, weights)
