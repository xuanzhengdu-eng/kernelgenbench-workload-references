"""Megatron torch-fallback KV-cache append.

Pinned source:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/inference/contexts/dynamic_context.py#L963-L1010

DynamicInferenceContext.append_key_value_cache takes this path when the fused Triton
helper is unavailable: it squeezes the singleton sequence axis and assigns
memory_buffer[0, layer, block_idx, local_pos] = key[:n] (and the value plane) with
advanced indexing.  layer_number is mapped through layer_map first.  Observable
behaviour matches the Triton branch for unique slots; the buffer is mutated in place
and the call returns None.
"""
import torch


def v1_store_bf16_kv_cache_megatron_torch(
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
