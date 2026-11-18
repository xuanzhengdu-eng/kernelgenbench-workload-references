"""vLLM multimodal per-channel affine transform for OP-368a31b88b4d.

Semantic reference (``forward_native``):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fusion/mm_input_norm.py#L345-L358
Layout unpacking (``pcl`` ``(P, C*L)`` and ``bchw`` ``(B, C, H, W)``):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fusion/mm_input_norm.py#L303-L340
Triton CUDA path (same ``input * weight[c] + bias[c]`` in fp32):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fusion/mm_input_norm.py#L33-L113

``weight``/``bias`` are the fp32 per-channel constants derived from
``rescale_factor / std`` and ``-mean / std``; the input is floating point or
uint8, the arithmetic is fp32, and the output is ``visual_dtype`` with the input
shape.  Inputs are read only.
"""
import torch


def v1_mm_channel_affine(pixel_values, weight, bias, visual_dtype):
    channels = weight.numel()
    if weight.ndim != 1 or bias.numel() != channels:
        raise ValueError("weight and bias must be 1-D tensors with equal length")
    if pixel_values.ndim == 4:
        if pixel_values.shape[1] != channels:
            raise ValueError("input has a different number of channels")
    elif pixel_values.ndim == 2:
        if pixel_values.shape[1] % channels:
            raise ValueError("flattened size is not divisible by the channel count")
    else:
        raise ValueError("pixel_values must be 2-D (P, C*L) or 4-D (B, C, H, W)")

    out_shape = pixel_values.shape
    x = pixel_values.reshape(pixel_values.shape[0], channels, -1)
    x = x * weight.view(1, -1, 1) + bias.view(1, -1, 1)
    return x.reshape(out_shape).to(visual_dtype)
