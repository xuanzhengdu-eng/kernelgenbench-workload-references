# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/layernorm_quant_kernels.cu#L160-L204
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/torch_bindings.cpp#L376-L384
import vllm._custom_ops
from torch.ops import _C

run = _C.fused_add_rms_norm_static_fp8_quant
