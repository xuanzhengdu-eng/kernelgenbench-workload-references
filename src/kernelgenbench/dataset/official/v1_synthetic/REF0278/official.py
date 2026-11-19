# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fused_moe/fused_globalize_align_block_size.py#L10-L63
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/moe/fused_globalize_align_block_size.cu#L28-L126
from vllm.model_executor.layers.fused_moe.fused_globalize_align_block_size import (
    fused_globalize_align_block_size as run,
)
