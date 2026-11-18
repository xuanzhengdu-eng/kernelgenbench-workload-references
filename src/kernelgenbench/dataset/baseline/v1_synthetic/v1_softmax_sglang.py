# Source-checked reference for KernelGenBench row 69 (SGLang branch).
"""SGLang sampling softmax with optional per-request temperature and output buffer.

``temperatures`` holds one value per request, each covering ``rows_per_request``
consecutive rows.  dtype follows torch promotion of ``logits / temperatures``;
``out`` is written in place when supplied (including ``out is logits``).

Official source (SGLang fixed commit 62384de1):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/sampling/__init__.py#L67-L105
"""
import torch


def v1_softmax_sglang(input, temperatures=None, rows_per_request=1, out=None):
    if temperatures is not None and rows_per_request > 1:
        temperatures = temperatures.repeat_interleave(rows_per_request, dim=0)
    logits = input / temperatures if temperatures is not None else input
    probs = torch.softmax(logits, dim=-1)
    return probs if out is None else out.copy_(probs)
