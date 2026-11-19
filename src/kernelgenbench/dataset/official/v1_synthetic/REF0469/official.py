# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/marlin_utils.py#L535-L547
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_moe_permute_scales as run,
)

__all__ = ["run"]
