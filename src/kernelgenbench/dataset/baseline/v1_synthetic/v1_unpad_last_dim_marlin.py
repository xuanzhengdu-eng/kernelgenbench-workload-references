"""vLLM Marlin: strip padded output columns back to the logical N.

Source (pinned commit f358f0578a83f4e1fe6fc49a492d3ae9362de8e5):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/marlin_utils.py#L288-L298

Transcribed from ``marlin_unpad_output``.  The padding decision is expressed
with the explicit ``padded_n`` argument: when it equals ``size_n`` the caller's
tensor is returned unchanged, otherwise the N prefix is sliced contiguously.
"""
import torch


def v1_unpad_last_dim_marlin(output, size_n, padded_n):
    if padded_n == size_n:
        return output
    return output[..., :size_n].contiguous()
