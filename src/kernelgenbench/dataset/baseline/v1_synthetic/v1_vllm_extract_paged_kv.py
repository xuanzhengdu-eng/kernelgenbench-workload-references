"""Gather token rows from a page-major cache for hidden-state transfer.

Official source (pinned vLLM commit):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py#L37-L46

``extract_from_kv_cache(kv_cache, slot_mapping, num_tokens)`` treats
``kv_cache`` as ``[num_blocks, num_heads, block_size, head_size]`` and returns
``kv_cache[slot // block_size, :, slot % block_size][:num_tokens]``.  The
advanced indexing produces a fresh tensor (never an alias of ``kv_cache``),
keeps the cache's dtype/device, and truncates to ``num_tokens`` rows, which may
be smaller than the slot list or zero.  Inputs are not modified.
"""


def v1_vllm_extract_paged_kv(kv_cache, slot_mapping, num_tokens):
    block_size = kv_cache.shape[2]
    return kv_cache[slot_mapping // block_size, :, slot_mapping % block_size][:num_tokens]
