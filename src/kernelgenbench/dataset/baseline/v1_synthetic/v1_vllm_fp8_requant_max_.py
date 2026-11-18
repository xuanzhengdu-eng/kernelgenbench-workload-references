"""vLLM max-scale requantization of fused FP8 weight shards.

https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/w8a8_utils.py#L75-L106
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/w8a8_utils.py#L44-L49
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/quantization/w8a8/fp8/common.cu#L263-L341

A fused FP8 weight carries one scale per logical shard. Each shard is
dequantized as float16 * float32 scale (float32 result), rescaled by the
reciprocal of the largest shard scale and written back into the same FP8
storage. Zero-width shards are skipped. A 0-dim scale means a checkpoint that
is already fused: the weight is untouched and only the scalar is returned.
Returns (max_scale, weight) where max_scale is the 0-dim float32 maximum and
weight is the same object that was passed in.
"""
import torch


def v1_vllm_fp8_requant_max_(weight, weight_scale, logical_widths):
    max_scale = weight_scale.max()
    if weight_scale.ndim != 0 and bool(weight_scale[-1] > torch.finfo(torch.float8_e4m3fn).min):
        inv = 1.0 / max_scale
        start = 0
        for index, width in enumerate(logical_widths):
            if width == 0:
                continue
            end = start + width
            dequantized = weight[start:end].to(torch.float16) * weight_scale[index]
            quantized = (dequantized * inv).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            weight[start:end].copy_(quantized)
            start = end
    return max_scale, weight
