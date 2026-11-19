# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/transformer/inference/op_binding/qkv_gemm.py#L60-L67
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/transformer/inference/op_binding/qkv_gemm.py#L70-L75
from deepspeed.ops.transformer.inference.op_binding.qkv_gemm import QKVGemmOp

qkv_gemm_fallback = QKVGemmOp.qkv_gemm_fallback
rms_qkv_gemm_fallback = QKVGemmOp.rms_qkv_gemm_fallback
