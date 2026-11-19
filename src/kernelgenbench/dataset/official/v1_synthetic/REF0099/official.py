# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fusion/mm_input_norm.py#L360-L369
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fusion/mm_input_norm.py#L69-L123
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fusion/mm_input_norm.py#L309-L338
from vllm.model_executor.layers.fusion.mm_input_norm import FusedMMInputNorm

run = FusedMMInputNorm.forward_cuda
norm_class = FusedMMInputNorm
