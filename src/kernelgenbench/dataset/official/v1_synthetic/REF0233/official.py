# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/lion/cpu_lion_impl.cpp#L192-L214
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/lion/cpu_lion_impl.cpp#L22-L65
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/lion/cpu_lion.py#L100-L128
from deepspeed.ops.op_builder import CPULionBuilder

run = CPULionBuilder().load().lion_update
