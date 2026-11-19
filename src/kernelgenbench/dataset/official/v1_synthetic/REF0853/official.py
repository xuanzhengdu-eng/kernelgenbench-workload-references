# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/attention/mla_attention.py#L409-L425
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/quant_utils.py#L528-L550
from vllm.model_executor.layers.attention.mla_attention import split_kv_b_proj as run
