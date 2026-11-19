# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/transformer/inference/op_binding/residual_add.py#L29-L51
from deepspeed.ops.transformer.inference.op_binding.residual_add import ResidualAddOp

run = ResidualAddOp.residual_add_fallback
