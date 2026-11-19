# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py#L296-L307
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/fp8_utils.py#L364-L469
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/fp8_utils.py#L472-L565
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    silu_mul_per_token_group_quant_fp8_colmajor as run,
)
