# Source-checked reference for KernelGenBench row 63.
"""Megatron batch-invariant affine RMSNorm forward (zero_centered_gamma=False).

FP32 statistics are accumulated by the batch-invariant Triton ``mean_dim``
kernel upstream; this reference keeps the same fp32 formula.  Official source:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/custom_layers/batch_invariant_kernels.py#L877-L945
"""
import torch


def v1_batch_rmsnorm(x, weight, eps=1e-5):
    x_dtype = x.dtype
    x_fp32 = x.float()
    w_fp32 = weight.to(device=x.device, dtype=torch.float32)
    rsigma = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
    return ((x_fp32 * rsigma) * w_fp32).to(x_dtype)
