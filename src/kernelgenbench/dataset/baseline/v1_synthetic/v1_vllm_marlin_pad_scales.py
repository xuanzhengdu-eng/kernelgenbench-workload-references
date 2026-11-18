"""Zero-pad GPTQ group scales to the Marlin tile extents.

https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/marlin_utils.py#L264-L278

Contract (official ``marlin_pad_scales``): ``scales`` is ``[num_groups, size_n]``
(a group row per ``group_size`` K elements). When ``(padded_n, padded_k)`` equals
``(size_n, size_k)`` the input is returned unchanged. Otherwise the number of
padded rows is ``padded_k / group_size - num_groups`` for a positive group size
and 0 for a channel-wise (``group_size <= 0``) weight; upstream asserts the row
delta is non-negative and pads width to ``padded_n``. A new zero tensor is
returned and the input is untouched.
"""
import torch


def v1_vllm_marlin_pad_scales(scales, size_n, size_k, padded_n, padded_k, group_size):
    if (padded_n, padded_k) == (size_n, size_k):
        return scales
    pad_rows = padded_k // group_size - scales.size(0) if group_size > 0 else 0
    assert pad_rows >= 0
    out = torch.zeros((scales.size(0) + pad_rows, scales.size(1) + padded_n - size_n),
                      device=scales.device, dtype=scales.dtype)
    out[: scales.size(0), : scales.size(1)] = scales
    return out
