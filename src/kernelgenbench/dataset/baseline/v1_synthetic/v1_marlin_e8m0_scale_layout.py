"""Marlin E8M0 scale layout for MXFP8 weight scales (vLLM).

Official source, exact code:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py#L346-L353

The MXFP8 path first reorders each group of four scale bytes from the
checkpoint order to the order the FP8 dequantization kernel reads
(``[0, 2, 1, 3]``), then reinterprets the FP32 bits as ``float8_e8m0fnu``.
Scales must be exact positive powers of two for the reinterpretation to be
lossless; the function does not add any exponent bias.  It returns a newly
allocated ``float8_e8m0fnu`` tensor with the same shape and leaves the input
untouched.

This is not the SGLang FP4->FP8 variant: SGLang's
``mxfp4_marlin_process_scales(..., input_dtype=float8_e4m3fn)`` keeps the
incoming byte order and adds 6 to every exponent, which is a different
contract and is carried by the separate item
``v1_sgl_mxfp4_e8m0_scale_layout``.
"""
import torch


def v1_marlin_e8m0_scale_layout(marlin_scales):
    reordered = marlin_scales.view(-1, 4)[:, [0, 2, 1, 3]].view(marlin_scales.shape)
    return reordered.to(torch.float8_e8m0fnu)
