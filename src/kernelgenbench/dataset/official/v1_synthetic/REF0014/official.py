# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/transformer/inference/op_binding/residual_add.py#L53-L73
from deepspeed.ops.transformer.inference.op_binding.residual_add import ResidualAddOp

run = ResidualAddOp.forward
