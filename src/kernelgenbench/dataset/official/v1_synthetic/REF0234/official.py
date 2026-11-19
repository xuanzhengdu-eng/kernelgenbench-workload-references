# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/runtime/base_optimizer.py#L428-L441
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/runtime/utils.py#L1332-L1342
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/runtime/fp16/loss_scaler.py#L165-L169
from deepspeed.runtime.base_optimizer import ZeROOptimizer
from deepspeed.runtime.fp16.loss_scaler import LossScaler

run = ZeROOptimizer.scale_if_loss
loss_scaler = LossScaler
