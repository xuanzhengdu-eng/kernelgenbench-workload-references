# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/w8a8_utils.py#L75-L106
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/w8a8_utils.py#L44-L49
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/quantization/w8a8/fp8/common.cu#L263-L341
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    requantize_with_max_scale as run,
)
