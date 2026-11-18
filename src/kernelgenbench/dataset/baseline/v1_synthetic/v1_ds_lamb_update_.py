"""DeepSpeed CUDA fused LAMB one-tensor update, official ``fused_lamb_cuda.lamb``.

https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/lamb/fused_lamb_cuda_kernel.cu#L194-L319
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/lamb/fused_lamb_cuda_kernel.cu#L321-L475

Contract reproduced from the pinned CUDA kernel:
* ``p`` is FP32, ``m``/``v`` are FP32, ``g`` is FP32 or FP16;
* ``m = beta1*m + (1-beta1)*(g/grad_scale)``;
* ``v = beta2*v + (1-beta2)*(g/grad_scale)^2``;
* ``denom = sqrt(v+eps)`` for ``mode == 0`` (eps inside sqrt) and
  ``sqrt(v)+eps`` for ``mode == 1``;
* ``update = m/denom + decay*p``;
* ``lamb_coeff = ||p|| / ||update||`` clamped to ``[min_coeff, max_coeff]``;
  when either norm is zero the kernel keeps ``1.0``;
* bias-corrected step size ``lr*sqrt(1-beta2**step)/(1-beta1**step)``, else ``lr``;
* ``p -= step_size * coeff * update`` and, only for an FP16 gradient with a
  non-empty ``p_copy``, ``p_copy.copy_(p)`` (cast to FP16);
* returns the FP32 coefficient as a one-element tensor.

The official kernel does not support BF16 gradients (``AT_DISPATCH_FLOATING_TYPES_AND_HALF``).
"""
import math

import torch


def v1_ds_lamb_update_(
    p,
    p_copy,
    m,
    v,
    g,
    lr,
    beta1,
    beta2,
    max_coeff,
    min_coeff,
    eps,
    grad_scale,
    step,
    mode,
    bias_correction,
    decay,
):
    scaled = g.float() / grad_scale
    m.mul_(beta1).add_(scaled, alpha=1 - beta1)
    v.mul_(beta2).addcmul_(scaled, scaled, value=1 - beta2)
    denom = (v + eps).sqrt() if mode == 0 else v.sqrt() + eps
    update = m / denom + decay * p
    wnorm = (p * p).sum().sqrt()
    unorm = (update * update).sum().sqrt()
    raw = wnorm / unorm
    coeff = torch.where(
        (wnorm != 0) & (unorm != 0),
        raw.clamp(min=min_coeff, max=max_coeff),
        torch.ones_like(raw),
    )
    step_size = lr * math.sqrt(1 - beta2**step) / (1 - beta1**step) if bias_correction else lr
    p.sub_(update * (step_size * coeff))
    if g.dtype == torch.float16 and p_copy.numel():
        p_copy.copy_(p)
    return coeff.reshape(1)
