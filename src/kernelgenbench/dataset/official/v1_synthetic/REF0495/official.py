# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/quantization/marlin/gptq_marlin_repack.cu#L15-L258
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/_custom_ops.py#L1110-L1116
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/_custom_ops.py#L1165-L1182
from vllm._custom_ops import gptq_marlin_repack as run
from vllm._custom_ops import gptq_marlin_moe_repack as run_moe
