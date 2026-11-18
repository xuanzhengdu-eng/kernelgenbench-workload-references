"""vLLM greedy argmax with its backend-dependent NaN rule.

https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/worker/gpu/sample/greedy_argmax.py#L8-L63

CUDA, 2-D, unit vocabulary stride, vocabulary >= 4096 and dtype float32 or
bfloat16 take the split Triton kernel, where NaN values are replaced by -inf.
Every other path calls ``torch.argmax``, which returns the NaN's index. Ties
resolve to the first index and the output is int64 ``[rows]``.
"""
import torch


def v1_greedy_argmax(logits):
    use_split = (
        logits.is_cuda
        and logits.dtype in (torch.bfloat16, torch.float32)
        and logits.ndim == 2
        and logits.stride(1) == 1
        and logits.shape[1] >= 4096
    )
    if use_split:
        logits = torch.where(torch.isnan(logits), float("-inf"), logits)
    return logits.argmax(dim=-1)
