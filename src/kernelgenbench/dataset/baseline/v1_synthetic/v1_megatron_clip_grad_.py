"""Megatron fp32 total-norm gradient clipping, official NVIDIA entry.

https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/optimizer/clip_grads.py#L145-L192

The legacy reference called the flagos-ai/Megatron-LM-FL fork and mixed two
unrelated branches: a Megatron local-fallback clip over ``p.grad`` and a
DeepSpeed ``clip_tensors_by_global_norm`` tensor-list variant.  The fork has no
official counterpart, and the DeepSpeed variant is the same official entry as
CSV row 175 ``v1_clip_tensors_global_norm_``, so this item is the Megatron
branch only and follows the pinned NVIDIA implementation:

* ``.grad`` must be a CUDA float tensor; ``.decoupled_grad`` may be fp32 or bf16.
* the coefficient is ``max_norm / (total_norm + 1e-6)``;
* gradients are scaled in place (same buffers) only when that coefficient is
  below one, and the function returns None.

``total_norm`` is a Python scalar: the official tensor-coefficient path needs
``multi_tensor_scale_tensor``, which the pinned site does not provide.
"""
import torch

def v1_megatron_clip_grad_(parameters, max_norm, total_norm, use_decoupled_grad=False):
    grads = []
    for param in parameters:
        if use_decoupled_grad:
            grad = getattr(param, "decoupled_grad", None)
            if grad is not None:
                assert grad.dtype in (torch.float32, torch.bfloat16)
                grads.append(grad.detach())
        else:
            grad = param.grad
            if grad is not None:
                assert grad.type() == "torch.cuda.FloatTensor"
                grads.append(grad.detach())
    clip_coeff = max_norm / (total_norm + 1.0e-6)
    if clip_coeff < 1.0:
        for grad in grads:
            grad.copy_(grad * clip_coeff)
