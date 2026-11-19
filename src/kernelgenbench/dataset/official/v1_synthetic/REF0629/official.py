# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/_custom_ops.py#L3252-L3270
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/cache_kernels.cu#L1050-L1150
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/cache_kernels.cu#L490-L536
from vllm._custom_ops import concat_and_cache_mla as run

__all__ = ["run"]
