"""Megatron SwiGLUFunction backward with an explicit saved-input context.

Official source (pinned commit 266f1c97dca477ca8b92c16087506da2000b0b84):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_bias_swiglu.py#L147-L188

Caller-visible contract: (grad, None, None). The saved input is restored from
FP8 to original_dtype when fp8_input_store is True before the elementwise
SwiGLU backward. No bias is involved; BiasSwiGLUFunction.backward is a separate
four-output contract and is not this item. The fork-only clamp_value mode has no
fixed upstream counterpart and is not part of this contract.
"""
import torch
import torch.nn.functional as F


def v1_swiglu_back_megatron_autograd(g, saved_input, original_dtype,
                                     fp8_input_store=False):
    y = saved_input.to(original_dtype) if fp8_input_store else saved_input
    y_1, y_2 = torch.chunk(y, 2, -1)
    grad = torch.cat(
        (g * torch.sigmoid(y_1) * (1 + y_1 * (1 - torch.sigmoid(y_1))) * y_2,
         g * F.silu(y_1)), -1)
    return grad, None, None
