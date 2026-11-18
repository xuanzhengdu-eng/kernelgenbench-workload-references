# Source-checked synthetic-workload benchmark reference.
"""OP-d2fa213d8170: vLLM quantized KV cache view. Non-FP4 inserts a dimension; FP4 builds an as_strided uint8 view with packed-data and scale bytes. Both alias original storage; input storage must cover requested strided view.

Source: https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/sparse_attn_indexer.py#L472-L489
"""
import torch

def v1_kv_cache_quant_view(kv_cache, head_dim, use_fp4_cache):
    if not use_fp4_cache: return kv_cache.unsqueeze(-2)
    assert kv_cache.ndim==3 and kv_cache.dtype==torch.uint8
    blocks,block_size,_=kv_cache.shape
    width=head_dim//2+head_dim//32
    return torch.as_strided(kv_cache,(blocks,block_size,1,width),(kv_cache.stride(0),width,width,1))
