# Source (limited scope): https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/compile/z3.cpp#L119-L164
# Scope: equal padded-shard gather, crop to true shape and dtype conversion as
# modelled by Z3CustomOpExecutor::allgatherParam plus register_z3_param metadata;
# the NCCL all-gather and its async events are outside this Python ABI model.
"""DeepSpeed ZeRO-3 equal padded shards: gather, crop padding and reshape.
"""
import math,torch,torch.distributed as dist
def v1_ds_zero3_gather_param(shard,true_shape,target_dtype,group=None):
    local=shard.to(dtype=target_dtype).contiguous().flatten()
    parts=[torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts,local,group=group)
    return torch.cat(parts)[:math.prod(true_shape)].view(true_shape)
