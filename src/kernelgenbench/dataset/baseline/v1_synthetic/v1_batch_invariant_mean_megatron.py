"""Megatron batch-invariant mean over one or more dimensions.

Official source (NVIDIA/Megatron-LM, commit 266f1c97dca477ca8b92c16087506da2000b0b84):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/custom_layers/batch_invariant_kernels.py#L496-L511
The single-dimension reduction it calls is ``mean_dim`` in the same file, lines 391-477.

Contract differs from vLLM: at least one dimension is required (``dim=()`` is not
accepted).  A single dimension uses the deterministic Triton ``mean_dim`` kernel, which
keeps the input dtype by default, so the ``dtype=torch.float32`` argument is validated
but *not* applied and an FP16 input still returns FP16.  Multi-dimension reductions
require a floating point input and always sum in FP32 then divide by the element count,
returning FP32.  Integer single-dimension input is reduced to FP32.
"""
import torch


def v1_batch_invariant_mean_megatron(input, dim, keepdim=False, dtype=None):
    assert dtype is None or dtype == torch.float32
    assert len(dim) > 0
    if len(dim) == 1:
        result = input.to(torch.float32).mean(dim=dim[0], keepdim=keepdim)
        return result.to(input.dtype) if input.is_floating_point() else result
    assert input.dtype in (torch.float16, torch.bfloat16, torch.float32)
    count = 1
    for d in dim:
        count *= input.shape[d]
    return torch.sum(input, dim=dim, keepdim=keepdim, dtype=torch.float32) / count
