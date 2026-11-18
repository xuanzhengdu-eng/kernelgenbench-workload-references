"""Reference: gather a multi-layer paged KV cache into ``[L, H, N, C]``.

Official implementation (pinned vLLM commit
``f358f0578a83f4e1fe6fc49a492d3ae9362de8e5``):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/utils/gather_scatter_helper.py#L167-L221

The upstream entry is ``gather_kv_caches``; it launches the Triton kernel
``kv_cache_gather_kernel`` with one program per ``(layer, token_position)``.
Per layer it reads the cache through the layer view's element strides
(``block_stride, head_stride, token_stride, content_stride``) at
``(index // tokens_per_block, :, index % tokens_per_block, :)`` and writes
``[layer, :, position, :]`` of ``dst_tensor``.  Token indices that are not
smaller than ``total_token_in_kvcache`` are skipped, so they keep the
destination's previous content.  The destination is updated in place and the
function returns ``None``; caches are only read.  Layer/head/content counts and
the destination shape come from the caller, and the upstream wrapper asserts
``dst_tensor.shape == (num_layers, num_heads, num_tokens, content_size)``.

This reference states that contract with plain tensor indexing, which is layout
independent and therefore also covers the non-contiguous (permuted) cache views
that the stride arguments exist for.
"""
import torch


def v1_vllm_hf3fs_gather_kv_(kv_caches, total_token_in_kvcache, dst_tensor,
                             token_indices, tokens_per_block):
    for layer, cache in enumerate(kv_caches):
        for position, index in enumerate(token_indices):
            if index >= total_token_in_kvcache:
                continue
            dst_tensor[layer, :, position].copy_(
                cache[index // tokens_per_block, :, index % tokens_per_block, :])
    return None
