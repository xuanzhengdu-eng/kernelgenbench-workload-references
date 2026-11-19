# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/mamba/ops/cpu/causal_conv1d.py#L119-L145
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/mamba/short_conv.py#L180-L186
from vllm.model_executor.layers.mamba.ops.cpu.causal_conv1d import causal_conv1d_update_torch as run
