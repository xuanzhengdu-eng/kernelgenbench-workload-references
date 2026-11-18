"""vLLM BF16 paged KV-cache store (Triton reshape-and-cache entry).

Pinned source:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py#L377-L482
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py#L32-L154

K/V rows are (tokens, heads, head_dim) and the caches are
(blocks, block_size, heads, head_dim).  Each non-negative slot writes
slot // block_size, slot % block_size; a negative slot marks a padded token and is
skipped.  Both K and V are required, the caches are mutated in place and the call
returns None.  This is the vLLM branch only; the SGLang and Megatron backends have
different signatures/shapes and live in their own reference files.
"""
import torch


def v1_store_bf16_kv_cache(
    kv, kv_cache, slot_mapping, value, value_cache
):
    assert value is not None and value_cache is not None
    assert kv.ndim == value.ndim == 3
    assert kv_cache.ndim == value_cache.ndim == 4
    valid = slot_mapping >= 0
    slots = slot_mapping[valid].long()
    block_size = kv_cache.shape[1]
    kv_cache[slots // block_size, slots % block_size] = kv[valid].reshape(
        -1, *kv_cache.shape[2:]
    )
    value_cache[slots // block_size, slots % block_size] = value[valid].reshape(
        -1, *value_cache.shape[2:]
    )
    return None
