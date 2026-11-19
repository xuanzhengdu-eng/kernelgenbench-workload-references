# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/conv.py#L110-L155
from vllm.model_executor.layers.conv import Conv2dLayer

run = Conv2dLayer.forward_cuda
