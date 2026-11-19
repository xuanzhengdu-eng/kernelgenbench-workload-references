# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/conv.py#L16-L100
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/conv.py#L160-L211
from vllm.model_executor.layers.conv import Conv3dLayer

forward_cuda = Conv3dLayer.forward_cuda
