# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/_custom_ops.py#L3213-L3249
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/gdn/fused_gdn_decode_kernel.cu#L150-L375
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/gdn/fused_gdn_decode_kernel.cu#L422-L597
from vllm._custom_ops import fused_gdn_decode_post_conv_mtp as run
