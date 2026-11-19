# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/transformer/inference/op_binding/gated_activation.py#L25-L37
from deepspeed.ops.transformer.inference.op_binding.gated_activation import GatedActivationOp

run = GatedActivationOp.gated_activation_fallback
