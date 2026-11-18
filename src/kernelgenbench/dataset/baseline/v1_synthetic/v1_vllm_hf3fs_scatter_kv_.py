"""Reference: scatter ``[L, H, N, C]`` rows into multi-layer paged KV caches.

Official implementation (pinned vLLM commit
``f358f0578a83f4e1fe6fc49a492d3ae9362de8e5``):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/utils/gather_scatter_helper.py#L110-L164

The upstream entry is ``scatter_kv_caches``; it launches the Triton kernel
``kv_cache_scatter_kernel`` with one program per ``(layer, token_position)``.
For every layer and token position ``p`` with token index ``i <
total_token_in_kvcache`` it writes ``src_tensor[layer, :, p, :]`` into the layer
cache view at ``(i // tokens_per_block, :, i % tokens_per_block, :)`` through
the view's four element strides.  Indices ``>= total_token_in_kvcache`` are
skipped and leave the cache untouched.  Caches are mutated in place, the source
tensor is only read, and the function returns ``None``.  The upstream wrapper
asserts ``src_tensor.shape == (num_layers, num_heads, num_tokens,
content_size)``.

This reference states the same contract with plain tensor indexing, which is
layout independent and therefore also covers the non-contiguous (permuted)
cache views that the stride arguments exist for.
"""
import torch


def v1_vllm_hf3fs_scatter_kv_(kv_caches, total_token_in_kvcache, src_tensor,
                              token_indices, tokens_per_block):
    for layer, cache in enumerate(kv_caches):
        for position, index in enumerate(token_indices):
            if index >= total_token_in_kvcache:
                continue
            cache[index // tokens_per_block, :, index % tokens_per_block, :].copy_(
                src_tensor[layer, :, position])
    return None
