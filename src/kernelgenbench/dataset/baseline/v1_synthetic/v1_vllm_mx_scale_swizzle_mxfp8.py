"""vLLM: swizzle MXFP8 per-block scales to the F8_128x4 layout.

Source (pinned commit f358f0578a83f4e1fe6fc49a492d3ae9362de8e5):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/mxfp8_utils.py#L16-L38

Transcribed from ``swizzle_mxfp8_scale``.  ``M`` rows and ``K`` values (one
scale per 32 values) are padded to 128 rows and to a multiple of four scale
columns, viewed as ``[mt, 4, 32, kt, 4]``, transposed to the
``[mt, kt, 32, 4, 4]`` swizzle and flattened.  Same layout and bytes as the
MXFP4 entry; the two call sites differ only in the consumer dtype.
"""
import torch


def v1_vllm_mx_scale_swizzle_mxfp8(sf, M, K):
    scaling_vector_size = 32
    factor = scaling_vector_size * 4
    num_m_tiles = (M + 127) // 128
    num_k_tiles = (K + factor - 1) // factor
    m_padded = num_m_tiles * 128
    k_scale_padded = num_k_tiles * 4
    scale_cols = K // scaling_vector_size
    sf_padded = torch.zeros((m_padded, k_scale_padded), dtype=sf.dtype,
                            device=sf.device)
    sf_padded[:M, :scale_cols] = sf
    sf_reshaped = sf_padded.view(num_m_tiles, 4, 32, num_k_tiles, 4)
    sf_swizzled = sf_reshaped.transpose(1, 3)
    return sf_swizzled.contiguous().view(-1)
