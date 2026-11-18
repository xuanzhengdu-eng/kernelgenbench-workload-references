"""Broadcast a shared rotary key to 128 heads and concatenate into `out` (SGLang).

Official pinned SGLang AOT CUDA kernel:
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/aot/csrc/elementwise/concat_mla.cu#L12-L126

Contract: `k_nope` [T, 128, 128] and `k_rope` [T, 1, 64] are bfloat16; `out` is
[T, 128, 192] bfloat16.  The last dimension of all three is unit stride; outer
strides may be padded and all pointers must be 16-byte aligned.  `out` is
written in place and None is returned.  The SGLang kernel is bfloat16-only and
requires exactly 128 heads.
"""
import torch


def v1_concat_mla_keys_sglang(k_nope, k_rope, out):
    assert k_nope.ndim == 3 and k_nope.shape[-1] == 128
    assert k_rope.ndim == 3 and k_rope.shape[1] == 1 and k_rope.shape[-1] == 64
    n = k_nope.shape[0]
    assert k_nope.shape[1] == 128 and out.ndim == 3
    assert k_rope.shape[0] == n and out.shape == (n, 128, 192)
    assert k_nope.dtype == k_rope.dtype == out.dtype == torch.bfloat16
    assert k_nope.device == k_rope.device == out.device
    assert k_nope.stride(-1) == k_rope.stride(-1) == out.stride(-1) == 1
    out[..., :128].copy_(k_nope)
    out[..., 128:].copy_(k_rope.expand(n, 128, 64))
    return None
