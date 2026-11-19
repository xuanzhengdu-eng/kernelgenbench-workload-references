# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/moe/moe_permute_unpermute_op.cu#L200-L214
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.cu#L130-L182
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fused_moe/moe_permute_unpermute.py#L90-L117
import torch

import vllm._moe_C_stable_libtorch

run = torch.ops._moe_C.moe_prepare_scatter
