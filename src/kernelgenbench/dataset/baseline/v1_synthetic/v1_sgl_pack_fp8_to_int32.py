"""Pack a two-dimensional E4M3 tensor into int32 words."""
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/quantization/marlin_utils_fp8.py#L196-L210

import torch


def v1_sgl_pack_fp8_to_int32(fp8_tensor, size_k_first=True):
    assert fp8_tensor.dtype == torch.float8_e4m3fn
    assert fp8_tensor.ndim == 2
    ordered = fp8_tensor.T if size_k_first else fp8_tensor
    packed = ordered.contiguous().view(torch.int32)
    return packed.T.contiguous() if size_k_first else packed
