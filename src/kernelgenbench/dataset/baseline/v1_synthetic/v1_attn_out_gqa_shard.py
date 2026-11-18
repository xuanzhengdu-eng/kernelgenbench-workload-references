"""DeepSpeed attention output weight/bias sharding for grouped-query attention.
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/v2/model_implementations/sharding/attn_out.py#L14-L64
"""
from .v1_shard_parameter import v1_shard_parameter

def v1_attn_out_gqa_shard(param,shard_rank,num_shards,head_size,n_heads_q,n_heads_kv):
    if param.ndim==1:return param if shard_rank==0 else None
    assert param.shape[0]==head_size*n_heads_q and n_heads_q!=n_heads_kv
    if n_heads_kv%num_shards==0:
        return v1_shard_parameter(param,'inner',shard_rank,num_shards,1,head_size,1)
    assert num_shards%n_heads_kv==0
    degree=num_shards//n_heads_kv;kvhead=shard_rank//degree;local=shard_rank%degree
    factor=n_heads_q//n_heads_kv
    chunk=param[...,factor*kvhead*head_size:factor*(kvhead+1)*head_size]
    return v1_shard_parameter(chunk,'inner',local,degree,1,head_size,1)
