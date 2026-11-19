# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fused_moe/deep_gemm_utils.py#L562-L576
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    deepgemm_unpermute_and_reduce as run,
)

__all__ = ["run"]
