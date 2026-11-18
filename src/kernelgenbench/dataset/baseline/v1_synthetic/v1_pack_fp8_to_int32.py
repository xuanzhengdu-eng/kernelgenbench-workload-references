"""Pack a two-dimensional E4M3 tensor into int32 words."""
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py#L331-L343

import torch


def v1_pack_fp8_to_int32(fp8_tensor, size_k_first=True):
    assert fp8_tensor.dtype == torch.float8_e4m3fn
    assert fp8_tensor.ndim == 2
    ordered = fp8_tensor.T if size_k_first else fp8_tensor
    packed = ordered.contiguous().view(torch.int32)
    return packed.T.contiguous() if size_k_first else packed
