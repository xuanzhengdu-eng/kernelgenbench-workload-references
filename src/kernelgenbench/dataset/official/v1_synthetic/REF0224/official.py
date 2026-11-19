# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/adagrad/cpu_adagrad.cpp#L20-L55
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/adagrad/cpu_adagrad.cpp#L178-L199
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/includes/cpu_adagrad.h#L65-L117
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/includes/simd.h#L34-L49
from deepspeed.ops.op_builder import CPUAdagradBuilder

_op = CPUAdagradBuilder().load()

run = _op.adagrad_update
create_adagrad = _op.create_adagrad
destroy_adagrad = _op.destroy_adagrad
