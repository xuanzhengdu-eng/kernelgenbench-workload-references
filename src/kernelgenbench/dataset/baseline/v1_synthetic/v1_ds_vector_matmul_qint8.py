"""DeepSpeed INT8 quantized weight dequantize + GEMM (``vector_matmul`` int8 mode).

Official sources (pinned commit ``8af59b2e``):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/transformer/inference/op_binding/vector_matmul.py#L12-L42
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/transformer/inference/csrc/pt_binding.cpp#L1331-L1380
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/transformer/inference/csrc/pt_binding.cpp#L1069-L1112

Contract of the ``q_int8`` argument of ``ds_vector_matmul<half>``: ``x`` is
FP16 ``[B, S, K]``, the weight is INT8 ``[O, K]`` laid out transposed for the
GEMM, and ``quant_scale`` is FP32 ``[G]`` with ``G`` groups over the output
rows.  The weight is dequantized per group with ``half(scale * code)`` and the
product is ``x @ dequant(weight).T`` computed by cuBLAS in FP16, returning a
fresh FP16 ``[B, S, O]`` tensor.  All tensors must be contiguous and on one
device; ``O % G == 0``.  The C++ entry needs the inference workspace to be
allocated for ``B*S*O`` elements, so the caller must allocate it first (the
Python wrapper does this at model setup).
"""
import torch


def v1_ds_vector_matmul_qint8(x, quantized_weight, quant_scale, group_size):
    weight = (quantized_weight.float()
              * quant_scale.repeat_interleave(group_size).unsqueeze(1)).half()
    return torch.matmul(x, weight.t())
