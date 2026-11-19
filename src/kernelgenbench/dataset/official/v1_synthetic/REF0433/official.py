# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/linear.py#L1832-L1859
from vllm.model_executor.layers.linear import RowParallelLinear

run = RowParallelLinear.forward
