# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fused_moe/moe_permute_unpermute.py#L285-L320
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.inl#L93-L166
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import moe_unpermute as run
