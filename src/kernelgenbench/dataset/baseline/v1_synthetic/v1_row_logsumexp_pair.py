"""SGLang row_logsumexp sub-contract from OP-d9a80ab59fb8 (723 row 54).

Official source (pinned commit 62384de1ae5e3bbd3159afd1639525b0bf39fbf9):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/logsumexp.py#L86-L110

Outputs two FP32 vectors (row_max, logsumexp_minus_row_max), not a single
logsumexp value. Contract kept from the old reference and the official kernel:
  * 2-D input only, fp16/bf16/fp32; the official function asserts both, so this
    reference asserts instead of raising ``ValueError``.
  * ``rows == 0`` returns two empty fp32 vectors; ``cols == 0`` returns
    ``(-inf, 0)`` per row.
  * fp32 accumulation, no mask, inputs are not modified.
"""
import torch


def v1_row_logsumexp_pair(x):
    assert x.ndim == 2, "row_logsumexp expects a 2-D tensor"
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32), \
        "x must be float16, bfloat16, or float32"
    rows, cols = x.shape
    if rows == 0:
        return (torch.empty((0,), dtype=torch.float32, device=x.device),
                torch.empty((0,), dtype=torch.float32, device=x.device))
    if cols == 0:
        return (torch.full((rows,), -torch.inf, dtype=torch.float32, device=x.device),
                torch.zeros((rows,), dtype=torch.float32, device=x.device))
    v = x.float()
    maxima = v.max(dim=-1).values
    log_sum = torch.logsumexp(v - maxima[:, None], dim=-1)
    return maxima, log_sum
