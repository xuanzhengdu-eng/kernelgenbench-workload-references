# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/triton_ops/group_gemm_triton.py#L430-L459
from deepspeed.ops.triton_ops.group_gemm_triton import _GroupGemmFn

run = _GroupGemmFn.backward
