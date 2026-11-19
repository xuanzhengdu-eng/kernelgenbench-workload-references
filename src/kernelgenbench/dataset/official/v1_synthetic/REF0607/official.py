# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/layers.py#L197-L303
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/vocab_parallel_embedding.py#L547-L569
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/vocab_parallel_embedding.py#L494-L529
from megatron.core.tensor_parallel.layers import VocabParallelEmbedding

forward = VocabParallelEmbedding.forward
