# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/pooler/seqwise/methods.py#L50-L57
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/pool/metadata.py#L120-L138
from vllm.model_executor.layers.pooler.seqwise.methods import LastPool

_last_pool = LastPool()

run = _last_pool.forward
