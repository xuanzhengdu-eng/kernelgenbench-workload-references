"""SGLang MLA absorbed-query concatenation, allocating result.

Official entry: ``concat_mla_absorb_q`` (JIT wrapper over the CUDA ConcatMlaAbsorbQKernel)
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/attention/concat_mla.py#L54-L72
Kernel: https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/aot/csrc/elementwise/concat_mla.cu#L123-L218

a [tokens, heads, 512] and b [tokens, heads, 64] must be bf16 (the CUDA kernel asserts
kBFloat16) with a contiguous last dimension; row/head strides may be padded and the
pointers must be 16-byte aligned. The wrapper allocates a fresh contiguous
[tokens, heads, 576] result, so unlike the old unified reference it takes no output
argument and returns the tensor. The vLLM preallocated-output entry is a separate item.
"""
import torch


def v1_concat_mla_query_sglang(a, b):
    assert a.ndim == b.ndim == 3 and a.shape[:2] == b.shape[:2]
    assert a.shape[-1] == 512 and b.shape[-1] == 64
    assert a.dtype == b.dtype == torch.bfloat16 and a.device == b.device
    assert a.shape[0] > 0 and a.shape[1] > 0
    assert a.stride(-1) == b.stride(-1) == 1
    out = torch.empty((*a.shape[:-1], a.shape[-1] + b.shape[-1]), dtype=a.dtype, device=a.device)
    out[..., :512].copy_(a)
    out[..., 512:].copy_(b)
    return out
