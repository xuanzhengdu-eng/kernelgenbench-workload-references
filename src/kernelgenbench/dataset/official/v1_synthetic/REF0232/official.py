# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/sequence/linear_cross_entropy.py#L215-L298
from deepspeed.sequence.linear_cross_entropy import vocab_parallel_linear_cross_entropy as run

__all__ = ["run"]
