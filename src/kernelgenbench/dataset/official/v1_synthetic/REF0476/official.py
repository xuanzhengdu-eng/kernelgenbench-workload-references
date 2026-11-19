# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py#L58-L122
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py#L36-L56
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    nvfp4_marlin_process_scales as run,
)

__all__ = ["run"]
