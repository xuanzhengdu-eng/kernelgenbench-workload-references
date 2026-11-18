"""DeepSpeed per-row (non-reduced) cross entropy over [rows, vocab] logits.

https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/runtime/chunked_cross_entropy.py#L159-L179
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/runtime/chunked_cross_entropy.py#L198-L239

Contract:
  * ``logits`` is ``[rows, vocab]`` in fp32/fp16/bf16; ``target`` is ``[rows]``
    int64. Logits are upcast to fp32 before ``logsumexp`` and the target gather.
  * ``reduction='none'`` returns one fp32 loss per row.
  * Rows whose target equals ``ignore_index`` return exactly zero (the gathered
    target is replaced by index 0 internally and the row is zeroed afterwards).
  * The result is a fresh tensor; neither input is modified.
  * No label smoothing parameter exists upstream; this entry exposes none.
"""
import torch.nn.functional as F


def v1_cross_entropy_rows(logits, target, ignore_index=-100):
    return F.cross_entropy(logits.float(), target, reduction='none',
                           ignore_index=ignore_index)
