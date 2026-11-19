# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/sparse_attn_indexer.py#L472-L489
from vllm.model_executor.layers.sparse_attn_indexer import kv_cache_as_quant_view as run

__all__ = ["run"]
