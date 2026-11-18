"""SGLang CUDA KV-cache store (store_cache JIT kernel).

Pinned source:
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/kvcache/kvcache.py#L69-L120

K/V are flat rows (tokens, heads * head_dim) and the caches are
(slots, heads * head_dim).  Writes whose slot equals reserved_skip_index are skipped
(0 is the CUDA-graph padding slot; pass -1 to disable).  The caches are mutated in
place and the call returns None.
"""
import torch


def v1_store_bf16_kv_cache_sgl_cuda(
    kv, value, kv_cache, value_cache, slot_mapping, reserved_skip_index=0
):
    assert value is not None and value_cache is not None
    assert kv.ndim == value.ndim == kv_cache.ndim == value_cache.ndim == 2
    keep = slot_mapping != reserved_skip_index
    slots = slot_mapping[keep].long()
    kv_cache[slots] = kv[keep]
    value_cache[slots] = value[keep]
    return None
