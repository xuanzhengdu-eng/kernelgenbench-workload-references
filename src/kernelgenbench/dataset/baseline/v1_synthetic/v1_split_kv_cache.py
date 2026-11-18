# Source-checked synthetic-workload benchmark reference.
"""OP-cce32e392f8b: DeepSpeed split_kv: a 5D [blocks,block_size,2,heads,head_size] cache returns two strided views into the original storage, not copies.

Source: https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/v2/ragged/kv_cache.py#L22-L37
"""
import torch

def v1_split_kv_cache(kv_cache):
    if kv_cache.ndim != 5: raise ValueError("KV cache must be 5D")
    return kv_cache[:,:,0,:,:],kv_cache[:,:,1,:,:]
