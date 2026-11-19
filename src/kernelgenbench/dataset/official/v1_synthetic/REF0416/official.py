# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/v2/model_implementations/sharding/qkv.py#L14-L128
from deepspeed.inference.v2.model_implementations.sharding.qkv import shard_qkv_param as run

__all__ = ["run"]
