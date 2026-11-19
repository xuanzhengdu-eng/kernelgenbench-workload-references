# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/activation.py#L477-L506
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/activation_kernels.cu#L1033-L1037
from vllm.model_executor.layers.activation import SwigluOAIAndMul

official_entry = SwigluOAIAndMul
