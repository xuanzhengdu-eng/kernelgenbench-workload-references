# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/triton_ops/swiglu_triton.py#L208-L229
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/triton_ops/swiglu_triton.py#L40-L54
from deepspeed.ops.triton_ops.swiglu_triton import swiglu as v1_swiglu_deepspeed_unpacked

__all__ = ["v1_swiglu_deepspeed_unpacked"]
