# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/transformer/inference/op_binding/softmax_context.py#L131-L149
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/transformer/inference/op_binding/workspace.py#L16-L129
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/transformer/inference/csrc/pt_binding.cpp#L439-L568
from deepspeed.ops.transformer.inference.op_binding.softmax_context import SoftmaxContextOp
from deepspeed.ops.transformer.inference.op_binding.workspace import InferenceContext

run = SoftmaxContextOp.forward

__all__ = ["run", "SoftmaxContextOp", "InferenceContext"]
