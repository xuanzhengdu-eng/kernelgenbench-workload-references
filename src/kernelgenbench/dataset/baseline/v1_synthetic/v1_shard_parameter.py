"""DeepSpeed aligned tensor-parallel parameter sharding without communication.
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/v2/model_implementations/sharding/utils.py#L13-L104
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/v2/model_implementations/sharding/embedding.py#L12-L21
"""
import torch

def v1_shard_parameter(param,mode,shard_rank,num_shards,num_concatenated_matrices=1,granularity=32,bias_dims=1):
    assert 0<=shard_rank<num_shards
    if param is None or num_shards==1:return param
    assert mode in ('inner','outer')
    axis=-1 if mode=='inner' or param.ndim==bias_dims else -2
    dim=param.shape[axis];assert dim%num_concatenated_matrices==0
    chunk=dim//num_concatenated_matrices;assert chunk%granularity==0
    blocks=chunk//granularity;base,extra=divmod(blocks,num_shards)
    start=(shard_rank*base+min(shard_rank,extra))*granularity
    end=start+(base+(shard_rank<extra))*granularity
    parts=param.split(chunk,dim=axis)
    return torch.cat([piece.narrow(axis,start,end-start) for piece in parts],dim=axis)
