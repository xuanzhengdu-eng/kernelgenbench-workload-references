"""Megatron fused Triton KV-cache append (triton_append_key_value_cache).

Pinned source:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/inference/contexts/fused_kv_append_kernel.py#L93-L187
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/inference/contexts/fused_kv_append_kernel.py#L21-L90

key/value are (tokens, 1, heads, head_dim) and memory_buffer is
(2, layers, blocks, block_size, heads, head_dim).  Only the first
padded_active_token_count token rows are written, at
memory_buffer[0/1, layer_number, token_to_block_idx[t], token_to_local_position[t]].
layer_number is the already-mapped cache plane.  The buffer is mutated in place and
the call returns None.
"""
import torch


def v1_store_bf16_kv_cache_megatron_triton(
    kv,
    value,
    memory_buffer,
    layer_number,
    padded_active_token_count,
    token_to_block_idx,
    token_to_local_position_within_kv_block,
):
    assert value is not None and memory_buffer is not None
    assert kv.ndim == value.ndim == 4 and kv.shape[1] == value.shape[1] == 1
    assert memory_buffer.ndim == 6
    n = int(padded_active_token_count)
    if n == 0:
        return None
    blocks = token_to_block_idx[:n].long()
    local = token_to_local_position_within_kv_block[:n].long()
    memory_buffer[0, layer_number, blocks, local] = kv[:n, 0]
    memory_buffer[1, layer_number, blocks, local] = value[:n, 0]
    return None
