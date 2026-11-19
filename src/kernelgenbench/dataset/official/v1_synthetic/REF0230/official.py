# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/lamb/fused_lamb_cuda_kernel.cu#L194-L319
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/lamb/fused_lamb_cuda_kernel.cu#L321-L475
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/lamb/fused_lamb_cuda.cpp#L14-L88
from deepspeed.ops.op_builder import FusedLambBuilder

run = FusedLambBuilder().load().lamb
