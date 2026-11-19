# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/transformer/inference/op_binding/residual_add.py#L53-L73
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/transformer/inference/op_binding/vector_add.py#L21-L28
from deepspeed.ops.transformer.inference.config import DeepSpeedInferenceConfig
from deepspeed.ops.transformer.inference.op_binding.residual_add import ResidualAddOp


def compute_official(hidden_state, residual, attention_output, mp_size):

    config = DeepSpeedInferenceConfig(hidden_size=hidden_state.shape[-1],
                                      max_out_tokens=max(1, hidden_state.shape[-2]))
    config.dtype = hidden_state.dtype
    config.mp_size = int(mp_size)
    config.pre_layer_norm = True
    config.mlp_after_attn = True
    op = ResidualAddOp(config)
    return op.forward(hidden_state, residual, False,
                      attention_output=attention_output,
                      attention_bias=None, final_bias=None)


v1_ds_attention_residual_no_bias = compute_official


def entry_identity():
    code = ResidualAddOp.forward.__code__
    return {
        "symbol": "deepspeed.ops.transformer.inference.op_binding.residual_add.ResidualAddOp.forward",
        "file": code.co_filename,
        "first_lineno": code.co_firstlineno,
        "vector_add": "deepspeed.ops.transformer.inference.op_binding.vector_add.VectorAddOp.forward",
    }
