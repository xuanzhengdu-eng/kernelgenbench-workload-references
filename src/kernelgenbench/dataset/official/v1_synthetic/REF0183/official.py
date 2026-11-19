# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/transformer/inference/csrc/pt_binding.cpp#L1069-L1112
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/transformer/inference/csrc/pt_binding.cpp#L1331-L1380
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/transformer/inference/op_binding/vector_matmul.py#L12-L42
import torch
from deepspeed.ops.transformer.inference.config import DeepSpeedInferenceConfig
from deepspeed.ops.transformer.inference.op_binding.vector_matmul import VectorMatMulOp


def run(x, quantized_weight, quant_scale, group_size=None):
    config = DeepSpeedInferenceConfig(
        hidden_size=quantized_weight.size(1),
        heads=1,
        num_hidden_layers=1,
        dtype=torch.float16,
        mp_size=1,
        use_triton=False,
        transposed_mode=False,
    )
    op = VectorMatMulOp(config)


    return op.vector_matmul_func(x, quantized_weight, False, quant_scale, True,
                                 config.transposed_mode)
