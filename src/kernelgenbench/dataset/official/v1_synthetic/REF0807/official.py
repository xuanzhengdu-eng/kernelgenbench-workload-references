# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/utils/tensor_cast.cpp#L7-L20
from deepspeed.ops.op_builder import UtilsBuilder

run = UtilsBuilder().load().cast_to_byte_tensor
