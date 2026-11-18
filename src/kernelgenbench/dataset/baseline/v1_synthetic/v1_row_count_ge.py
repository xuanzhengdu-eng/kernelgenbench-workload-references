"""Count row elements greater than or equal to the per-row value.

Official source (vLLM main, commit f358f0578a83f4e1fe6fc49a492d3ae9362de8e5):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/sample/ops/logprobs.py#L10-L27

The upstream docstring says "greater than", but its computation is ``(x >= values)``,
so ties are counted.  ``x`` is 2-D ``(batch, n)`` and ``values`` is ``(batch, 1)``;
the result is int64 ``(batch,)``.  ``n == 0`` yields zero for every row.  The upstream
function is wrapped in ``torch.compile``; this reference is the same eager expression.
"""
import torch


def v1_row_count_ge(x, values):
    assert x.ndim == 2 and values.shape == (x.shape[0], 1)
    assert x.shape[0] >= 1
    return (x >= values).sum(-1)
