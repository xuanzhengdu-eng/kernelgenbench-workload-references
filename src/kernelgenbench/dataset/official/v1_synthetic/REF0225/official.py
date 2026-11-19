# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/adam/cpu_adam_impl.cpp#L35-L87
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/adam/cpu_adam_impl.cpp#L251-L277
from deepspeed.ops.op_builder import CPUAdamBuilder

adam_update = CPUAdamBuilder().load().adam_update
