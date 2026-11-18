"""Zero-pad a GPTQ-layout packed weight to the Marlin tile extents.

https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/marlin_utils.py#L251-L261

Contract (official ``marlin_pad_qweight``): ``qweight`` is int32
``[size_k / pack_factor, size_n]`` with ``pack_factor = size_k // qweight.size(0)``.
When ``(padded_n, padded_k)`` already equals ``(size_n, size_k)`` the input is
returned unchanged (same object). Otherwise a new zero tensor of
``[rows + (padded_k - size_k) / pack_factor, padded_n]`` is allocated and the
input is copied into its top-left corner. The input is never modified.
"""
import torch


def v1_vllm_marlin_pad_qweight(qweight, size_n, size_k, padded_n, padded_k):
    if (padded_n, padded_k) == (size_n, size_k):
        return qweight
    pack_factor = size_k // qweight.size(0)
    rows = qweight.size(0) + (padded_k - size_k) // pack_factor
    out = torch.zeros((rows, qweight.size(1) + padded_n - size_n), device=qweight.device, dtype=qweight.dtype)
    out[: qweight.size(0), : qweight.size(1)] = qweight
    return out
