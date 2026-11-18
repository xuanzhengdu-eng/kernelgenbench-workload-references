"""vLLM MLA query concatenation into a preallocated output.

Official entry: ``concat_mla_q`` (``torch.ops._C_cache_ops.concat_mla_q``)
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/cache_kernels.cu#L2168-L2209

ql_nope [tokens, heads, nope_dim] with nope_dim a multiple of 512, q_pe
[tokens, heads, 64] and q_out [tokens, heads, nope_dim+64]; all fp16 or bf16 with a
contiguous last dimension, padded row/head strides allowed. The kernel writes q_out in
place and returns nothing, so the output pointer and any trailing guard columns are part
of the contract. The SGLang allocating entry is a separate item.
"""
import torch


def v1_concat_mla_query_vllm(ql_nope, q_pe, q_out):
    assert ql_nope.ndim == q_pe.ndim == q_out.ndim == 3
    nope_dim, rope_dim = ql_nope.shape[-1], q_pe.shape[-1]
    assert nope_dim % 512 == 0 and rope_dim == 64
    assert q_out.shape[-1] == nope_dim + rope_dim
    assert ql_nope.shape[:2] == q_pe.shape[:2] == q_out.shape[:2]
    assert ql_nope.shape[0] > 0 and ql_nope.shape[1] > 0
    assert ql_nope.dtype == q_pe.dtype == q_out.dtype
    assert ql_nope.dtype in (torch.float16, torch.bfloat16)
    assert ql_nope.device == q_pe.device == q_out.device
    assert ql_nope.stride(-1) == q_pe.stride(-1) == q_out.stride(-1) == 1
    q_out[..., :nope_dim].copy_(ql_nope)
    q_out[..., nope_dim:].copy_(q_pe)
    return None
