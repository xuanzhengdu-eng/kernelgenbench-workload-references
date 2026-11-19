# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/aot/csrc/cpu/norm.cpp#L1161-L1299
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/aot/csrc/cpu/norm.cpp#L1371-L1436
import torch

run = torch.ops.sgl_kernel.fused_qk_norm_rope_cpu
