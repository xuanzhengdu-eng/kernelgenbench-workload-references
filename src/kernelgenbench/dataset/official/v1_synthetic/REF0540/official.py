# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/cpu/sgl-kernels/conv.cpp#L543-L585
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/cpu/torch_bindings.cpp#L602-L604
from vllm._custom_ops import causal_conv1d_weight_pack as run

__all__ = ["run"]
