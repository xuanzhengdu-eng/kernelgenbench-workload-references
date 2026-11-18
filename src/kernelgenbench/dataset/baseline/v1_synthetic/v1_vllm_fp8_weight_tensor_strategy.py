"""vLLM per-tensor FP8 strategy: requantize fused shards and return ``(K, N)``.

Source (pinned commit f358f0578a83f4e1fe6fc49a492d3ae9362de8e5):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/fp8_utils.py#L1454-L1480

The body is transcribed from ``process_fp8_weight_tensor_strategy`` and its
helpers ``requantize_with_max_scale`` / ``_maybe_pad_fp8_weight``.  The ROCm
FNUZ normalization and the ROCm padding branch are kept as platform gates; on
CUDA/H800 both are inactive.  ``ops.scaled_fp8_quant`` is replaced by the
equivalent elementwise quantize step it performs for a static per-tensor scale
(dq * 1/max_scale, clamp to +-448, cast to E4M3FN); bitwise equality with the
real kernel is asserted by the companion test.
"""
import os

import torch


def _is_fp8_fnuz():
    """ROCm reports the fp8 dtype as E4M3FNUZ; CUDA/H800 does not."""
    return torch.version.hip is not None


def _normalize_e4m3fn_to_e4m3fnuz(weight, weight_scale, input_scale=None):
    assert weight.dtype == torch.float8_e4m3fn
    weight_as_int8 = weight.view(torch.int8)
    weight_as_int8[weight_as_int8 == -128] = 0
    weight = weight_as_int8.view(torch.float8_e4m3fnuz)
    weight_scale = weight_scale * 2.0
    if input_scale is not None:
        input_scale = input_scale * 2.0
    return weight, weight_scale, input_scale


def _maybe_pad_fp8_weight(weight):
    # ROCm-only padding optimization; inactive on CUDA.
    if (os.environ.get("VLLM_ROCM_FP8_PADDING", "1") == "1"
            and torch.version.hip is not None
            and weight.stride(-1) == 1
            and (weight.stride(-2) * weight.element_size()) % 512 == 0):
        num_pad = 256 // weight.element_size()
        weight = torch.nn.functional.pad(weight, (0, num_pad), "constant", 0)[..., :-num_pad]
    return weight


def _requantize_with_max_scale(weight, weight_scale, logical_widths):
    max_w_scale = weight_scale.max()
    unfused_module_in_checkpoint = (
        weight_scale.ndim != 0
        and weight_scale[-1] > torch.finfo(torch.float8_e4m3fn).min
    )
    if unfused_module_in_checkpoint:
        start = 0
        for idx, logical_width in enumerate(logical_widths):
            if logical_width == 0:
                continue
            end = start + logical_width
            weight_dq = weight[start:end, :].to(torch.float16) * weight_scale[idx]
            quantized = (weight_dq.float() * max_w_scale.reciprocal()).clamp(
                torch.finfo(torch.float8_e4m3fn).min,
                torch.finfo(torch.float8_e4m3fn).max,
            ).to(weight.dtype)
            weight[start:end, :] = quantized
            start = end
    return max_w_scale, weight


def v1_vllm_fp8_weight_tensor_strategy(weight, weight_scale, logical_widths,
                                       input_scale=None):
    if _is_fp8_fnuz() and weight.dtype == torch.float8_e4m3fn:
        weight, weight_scale, input_scale = _normalize_e4m3fn_to_e4m3fnuz(
            weight=weight, weight_scale=weight_scale, input_scale=input_scale)
    weight_scale, weight = _requantize_with_max_scale(
        weight=weight, weight_scale=weight_scale, logical_widths=logical_widths)
    weight = _maybe_pad_fp8_weight(weight).t()
    return weight, weight_scale, input_scale
