# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/transformer/inference/csrc/gelu.cu#L145-L289
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/transformer/inference/csrc/gelu.cu#L300-L423
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/transformer/inference/csrc/pt_binding.cpp#L1815-L1851
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/transformer/inference/op_binding/residual_add.py#L16-L44
from deepspeed.ops.transformer.inference.op_binding.residual_add import ResidualAddOp


def run(op: ResidualAddOp, hidden_state, residual, attention_output, attention_bias,
        final_bias, mp_size, mlp_after_attn, add_attn_bias, pre_attn_norm):







    return op.residual_add_func(hidden_state, residual, attention_output,
                                attention_bias, final_bias, mp_size,
                                mlp_after_attn, add_attn_bias, pre_attn_norm)
