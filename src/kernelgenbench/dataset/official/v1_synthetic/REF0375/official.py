# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/sparse_attn_indexer.py#L274-L315
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/models/deepseek_v2.py#L797-L805
from vllm.model_executor.layers.sparse_attn_indexer import (
    fused_indexer_q_rope_quant as run,
)
