"""Megatron Mamba attention context-parallel load balancing (reorder / undo).

Official NVIDIA fixed commit:
  * ``_redo_attention_load_balancing``: for cp_size=3 the sequence becomes
    1 6 2 5 3 4 -- chunk pairs (i, 2*cp_size-1-i) are interleaved so every rank
    holds one high and one low chunk.
  * ``_undo_attention_load_balancing``: the inverse order 1 2 3 4 5 6, built from
    the even chunks followed by the reversed odd chunks.
Both operate on ``torch.chunk(input_, 2*cp_size, dim=0)``; a one-rank group still
returns a fresh concatenation.
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/ssm/mamba_context_parallel.py#L379-L414
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/ssm/mamba_context_parallel.py#L417-L453

Packed (THD) sequences use Transformer Engine's ``thd_get_partitioned_indices`` in
the official path.  Transformer Engine is not importable in the pinned containers,
so the packed branch here takes an explicit ``packed_indices`` plan instead and is a
documented adapter, not the native path.
"""
import torch
def v1_mamba_cp_rebalance(input_,cp_size,inverse,packed_indices=None):
    if packed_indices is not None:
        idx=packed_indices.reshape(-1).to(device=input_.device,dtype=torch.int64)
        if not inverse:return input_.index_select(0,idx)
        out=torch.empty_like(input_);out[idx]=input_;return out
    chunks=torch.chunk(input_,2*cp_size,dim=0)
    order=([2*i for i in range(cp_size)]+[2*cp_size-2*i-1 for i in range(cp_size)]) if inverse else [v for i in range(cp_size) for v in (i,2*cp_size-i-1)]
    return torch.cat([chunks[i] for i in order],dim=0)
