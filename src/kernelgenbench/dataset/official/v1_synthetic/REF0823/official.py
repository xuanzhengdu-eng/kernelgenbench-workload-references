# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/aot/csrc/cpu/model/qwen3.cpp#L200-L241
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/aot/csrc/cpu/model/qwen3.cpp#L67-L97
import torch

run = torch.ops.sgl_kernel.fused_qkvzba_split_reshape_cat_contiguous_cpu
