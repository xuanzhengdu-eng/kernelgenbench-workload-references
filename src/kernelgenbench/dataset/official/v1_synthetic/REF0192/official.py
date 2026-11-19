# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fused_qk_norm_rope.py#L139-L280
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fused_qk_norm_rope.py#L17-L136
from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate as run
