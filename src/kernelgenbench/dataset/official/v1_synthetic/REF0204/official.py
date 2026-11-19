# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/models/qwen3_vl.py#L258-L301
from vllm.model_executor.models.qwen3_vl import triton_pos_embed_interpolate as run

# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/models/qwen3_vl.py#L304-L374
from vllm.model_executor.models.qwen3_vl import (
    pos_embed_interpolate_native as run_native_formula,
)
