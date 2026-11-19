# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/activation.py#L666-L668
from vllm.model_executor.layers.activation import ReLUSquaredActivation



run = ReLUSquaredActivation.forward_native
