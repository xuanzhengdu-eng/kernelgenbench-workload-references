# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fused_moe/utils.py#L458-L503
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fused_moe/utils.py#L506-L593

from vllm.model_executor.layers.fused_moe.utils import masked_amax_scale as run

__all__ = ["run"]
