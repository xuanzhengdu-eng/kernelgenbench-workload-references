# Source-checked synthetic-workload benchmark reference.
"""OP-c570f91297d7: Megatron batch-invariant RMSNorm explicit backward,
zero_centered_gamma=False.

Official source (pinned commit 266f1c97dca477ca8b92c16087506da2000b0b84):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/custom_layers/batch_invariant_kernels.py#L912-L935

Contract kept from the old reference and the official backward:
  * ``x`` and ``grad_output`` have the same shape; ``weight`` is 1-D over the
    last axis; ``rsigma`` is the saved fp32 ``rsqrt(mean(x^2) + eps)`` with a
    trailing singleton dimension.
  * all arithmetic runs in fp32; the weight gradient is reduced over every
    leading dimension and cast to ``weight.dtype``; ``dx`` is cast to ``x.dtype``.
  * the source formula keeps an extra weight factor in the second dx term
    (``(w * r^3) * (s * x) / D``), which differs from the ordinary RMSNorm
    derivative; it is preserved verbatim.
  * returns ``(dx, g_w, None, None)``, matching the upstream autograd backward
    tuple (no gradient for eps or zero_centered_gamma).
"""
import torch


def v1_batch_rmsnorm_back(grad_output, x, weight, rsigma):
    g = grad_output.float()
    v = x.float()
    w = weight.float()
    r = rsigma
    g_w = (g * v * r).sum(dim=tuple(range(g.ndim - 1))).to(weight.dtype)
    s = (g * v * w).sum(dim=-1, keepdim=True)
    dx = g * (w * r) - (w * r * r * r) * (s * v) / x.shape[-1]
    return dx.to(x.dtype), g_w, None, None
